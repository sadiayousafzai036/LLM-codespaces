"""Build the searchable knowledge base from the raw dlt tables.

One document is one question paired with one of its answers. That granularity
was chosen over "one document per question" because the answer text is what
actually contains the advice, and over "one document per paragraph" because
Stack Exchange answers are already short and self-contained - splitting them
mid-technique produced chunks that read as instructions with no subject.

We keep at most MAX_ANSWERS_PER_QUESTION answers per question: the accepted one
plus the next highest voted. Beyond that, answers tend to repeat each other and
just crowd out other questions in the retrieved context.
"""

from ingestion.html_text import html_to_text, truncate
from app import db

# Long enough to keep a full technique explanation, short enough that five
# documents still leave room in the context window.
MAX_ANSWER_CHARS = 2500
MAX_QUESTION_CHARS = 1200

MAX_ANSWERS_PER_QUESTION = 2

# Below this, an "answer" is usually a one-line "me too" or a bare link.
MIN_ANSWER_CHARS = 80

JOIN_SQL = """
SELECT
    q.question_id,
    q.title,
    q.body_html          AS question_html,
    q.tags,
    q.primary_tag,
    q.score              AS question_score,
    q.link,
    q.owner_display_name AS question_author,
    a.answer_id,
    a.body_html          AS answer_html,
    a.score              AS answer_score,
    a.is_accepted,
    a.owner_display_name AS answer_author
FROM raw.questions AS q
JOIN raw.answers   AS a ON a.question_id = q.question_id
ORDER BY q.score DESC, q.question_id, a.is_accepted DESC, a.score DESC
"""


def _attribution(row: dict) -> str:
    """Stack Exchange content is CC BY-SA and requires author attribution."""
    author = row.get("answer_author") or "anonymous"
    asker = row.get("question_author") or "anonymous"
    link = row.get("link") or "link unavailable"
    return f"Answer by {author}, question by {asker} ({link})"


def _to_document(row: dict) -> dict | None:
    answer_text = html_to_text(row["answer_html"] or "")
    if len(answer_text) < MIN_ANSWER_CHARS:
        return None

    title = html_to_text(row["title"] or "").strip()
    question_text = html_to_text(row["question_html"] or "")

    return {
        "doc_id": f"q{row['question_id']}-a{row['answer_id']}",
        "question_id": row["question_id"],
        "answer_id": row["answer_id"],
        "title": title,
        "question_text": truncate(question_text, MAX_QUESTION_CHARS),
        "answer_text": truncate(answer_text, MAX_ANSWER_CHARS),
        "tags": row["tags"] or "",
        "primary_tag": row["primary_tag"] or "untagged",
        "question_score": int(row["question_score"] or 0),
        "answer_score": int(row["answer_score"] or 0),
        "is_accepted": bool(row["is_accepted"]),
        "link": row["link"] or "",
        "attribution": _attribution(row),
    }


def build_documents() -> list[dict]:
    for table in ("questions", "answers"):
        if not db.table_exists("raw", table):
            raise RuntimeError(
                f"raw.{table} is missing - run the extraction step first "
                f"(python -m ingestion.run)"
            )

    rows = db.fetch_all(JOIN_SQL)
    print(f"read {len(rows)} question/answer pairs from the raw schema")

    documents: list[dict] = []
    kept_per_question: dict[int, int] = {}
    skipped_short = 0

    # The SQL already orders accepted answers first, then by score, so taking
    # the first N per question gives us the best ones.
    for row in rows:
        question_id = row["question_id"]
        if kept_per_question.get(question_id, 0) >= MAX_ANSWERS_PER_QUESTION:
            continue

        document = _to_document(row)
        if document is None:
            skipped_short += 1
            continue

        documents.append(document)
        kept_per_question[question_id] = kept_per_question.get(question_id, 0) + 1

    print(
        f"built {len(documents)} documents "
        f"covering {len(kept_per_question)} questions "
        f"({skipped_short} answers skipped as too short)"
    )
    return documents


def rebuild() -> int:
    db.init_schema()
    documents = build_documents()
    if not documents:
        raise RuntimeError("no documents were built - is the raw schema empty?")
    db.replace_documents(documents)
    print(f"wrote {len(documents)} rows to public.documents")
    return len(documents)


if __name__ == "__main__":
    rebuild()
