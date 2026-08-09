"""Postgres access: schema, writes and the reads the dashboard needs.

One database holds three things, in two schemas:

  raw.questions, raw.answers   loaded by the dlt pipeline, untouched API payloads
  public.documents             the cleaned knowledge base the app searches
  public.conversations         one row per answered question
  public.feedback              user thumbs and LLM-judge verdicts

Keeping the raw layer separate means we can rebuild the knowledge base without
spending API quota again, which matters because the anonymous Stack Exchange
quota is only 300 requests a day.
"""

import time
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from app import settings

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id          TEXT PRIMARY KEY,
        question_id     BIGINT NOT NULL,
        answer_id       BIGINT NOT NULL,
        title           TEXT NOT NULL,
        question_text   TEXT NOT NULL,
        answer_text     TEXT NOT NULL,
        tags            TEXT NOT NULL,
        primary_tag     TEXT NOT NULL,
        question_score  INTEGER NOT NULL,
        answer_score    INTEGER NOT NULL,
        is_accepted     BOOLEAN NOT NULL,
        link            TEXT NOT NULL,
        attribution     TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS documents_primary_tag_idx ON documents (primary_tag)",
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id                BIGSERIAL PRIMARY KEY,
        question          TEXT NOT NULL,
        answer            TEXT NOT NULL,
        tag_filter        TEXT,
        strategy          TEXT NOT NULL,
        model             TEXT NOT NULL,
        prompt_variant    TEXT NOT NULL,
        rewritten_query   TEXT,
        retrieved_doc_ids TEXT NOT NULL,
        top_tag           TEXT,
        prompt_tokens     INTEGER NOT NULL,
        completion_tokens INTEGER NOT NULL,
        total_tokens      INTEGER NOT NULL,
        retrieval_time    DOUBLE PRECISION NOT NULL,
        response_time     DOUBLE PRECISION NOT NULL,
        cost              DOUBLE PRECISION NOT NULL,
        created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS conversations_created_at_idx ON conversations (created_at)",
    """
    CREATE TABLE IF NOT EXISTS feedback (
        id              BIGSERIAL PRIMARY KEY,
        conversation_id BIGINT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
        source          TEXT NOT NULL,
        score           INTEGER,
        relevance       TEXT,
        groundedness    TEXT,
        explanation     TEXT,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS feedback_conversation_idx ON feedback (conversation_id)",
]


@contextmanager
def connect():
    """Connection that commits on clean exit and rolls back on error."""
    conn = psycopg.connect(settings.postgres_dsn(), row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> None:
    with connect() as conn, conn.cursor() as cur:
        for statement in DDL_STATEMENTS:
            cur.execute(statement)


def wait_until_ready(timeout: float = 60.0, interval: float = 2.0) -> None:
    """Block until Postgres accepts connections.

    Compose healthchecks cover the containers, but the same code runs locally
    right after `make db`, where the server needs a few seconds to come up.
    """
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
            return
        except psycopg.OperationalError as exc:
            last_error = exc
            print("waiting for postgres...")
            time.sleep(interval)
    raise RuntimeError(f"postgres did not become ready in {timeout}s") from last_error


def fetch_all(sql: str, params: tuple | None = None) -> list[dict]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def fetch_one(sql: str, params: tuple | None = None) -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def table_exists(schema: str, table: str) -> bool:
    row = fetch_one(
        """
        SELECT 1 AS present
        FROM information_schema.tables
        WHERE table_schema = %s AND table_name = %s
        """,
        (schema, table),
    )
    return row is not None


# --------------------------------------------------------------- documents


def replace_documents(documents: list[dict]) -> None:
    """Swap in a freshly built knowledge base as a single transaction.

    The knowledge base is a snapshot of the top-voted posts, so a full replace
    is the honest operation. Doing it inside one transaction means the app never
    observes a half-empty table.
    """
    columns = [
        "doc_id", "question_id", "answer_id", "title", "question_text",
        "answer_text", "tags", "primary_tag", "question_score", "answer_score",
        "is_accepted", "link", "attribution",
    ]
    placeholders = ", ".join(["%s"] * len(columns))
    insert_sql = (
        f"INSERT INTO documents ({', '.join(columns)}) VALUES ({placeholders})"
    )

    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM documents")
        cur.executemany(
            insert_sql,
            [tuple(doc[column] for column in columns) for doc in documents],
        )


def load_documents(tag: str | None = None) -> list[dict]:
    sql = "SELECT * FROM documents"
    params: tuple = ()
    if tag:
        sql += " WHERE primary_tag = %s"
        params = (tag,)
    sql += " ORDER BY doc_id"
    return fetch_all(sql, params)


def list_tags(limit: int = 40) -> list[str]:
    rows = fetch_all(
        """
        SELECT primary_tag, COUNT(*) AS n
        FROM documents
        GROUP BY primary_tag
        ORDER BY n DESC, primary_tag
        LIMIT %s
        """,
        (limit,),
    )
    return [row["primary_tag"] for row in rows]


def count_documents() -> int:
    row = fetch_one("SELECT COUNT(*) AS n FROM documents")
    return row["n"] if row else 0


# ----------------------------------------------------------- conversations


def save_conversation(record: dict) -> int:
    columns = [
        "question", "answer", "tag_filter", "strategy", "model",
        "prompt_variant", "rewritten_query", "retrieved_doc_ids", "top_tag",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "retrieval_time", "response_time", "cost",
    ]
    placeholders = ", ".join(["%s"] * len(columns))
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO conversations ({', '.join(columns)}) "
            f"VALUES ({placeholders}) RETURNING id",
            tuple(record.get(column) for column in columns),
        )
        return cur.fetchone()["id"]


def save_feedback(
    conversation_id: int,
    source: str,
    score: int | None = None,
    relevance: str | None = None,
    groundedness: str | None = None,
    explanation: str | None = None,
) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO feedback (
                conversation_id, source, score, relevance, groundedness, explanation
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (conversation_id, source, score, relevance, groundedness, explanation),
        )


def user_feedback_for(conversation_id: int) -> int | None:
    """The thumb the user already gave, so the UI can show it as chosen."""
    row = fetch_one(
        """
        SELECT score
        FROM feedback
        WHERE conversation_id = %s AND source = 'user'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (conversation_id,),
    )
    return row["score"] if row else None
