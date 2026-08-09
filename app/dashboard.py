"""Streamlit monitoring dashboard.

    uv run streamlit run app/dashboard.py --server.port 8502

Eight charts over the two logging tables. The SQL lives here rather than in
db.py on purpose: each query exists only to feed the chart directly below it, so
keeping them together makes the panel easy to change.

The time-bucketed charts group by hour. A demo database holds a few dozen rows
spread over an evening, and grouping by day would collapse all of it into one
bar.
"""

import altair as alt
import pandas as pd
import streamlit as st

from app import db, settings

st.set_page_config(page_title="Cooking Assistant Dashboard", page_icon="📊",
                   layout="wide")

REFRESH_SECONDS = 60

RELEVANCE_ORDER = ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]
GROUNDEDNESS_ORDER = ["SUPPORTED", "PARTLY_SUPPORTED", "UNSUPPORTED"]

# Green through amber to red, so a good result reads as good at a glance.
VERDICT_COLOURS = ["#2e7d32", "#f9a825", "#c62828"]


@st.cache_data(ttl=REFRESH_SECONDS)
def frame(sql: str, params: tuple | None = None) -> pd.DataFrame:
    return pd.DataFrame(db.fetch_all(sql, params))


def bar(data: pd.DataFrame, x: str, y: str, x_title: str, y_title: str,
        colour: str = "#4c78a8") -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_bar(color=colour, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X(x, title=x_title),
            y=alt.Y(y, title=y_title),
            tooltip=list(data.columns),
        )
        .properties(height=260)
    )


def ordered_bar(data: pd.DataFrame, category: str, order: list[str],
                title: str) -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=2)
        .encode(
            x=alt.X("count:Q", title="answers"),
            y=alt.Y(f"{category}:N", sort=order, title=None),
            color=alt.Color(
                f"{category}:N",
                scale=alt.Scale(domain=order, range=VERDICT_COLOURS),
                legend=None,
            ),
            tooltip=[category, "count"],
        )
        .properties(height=200, title=title)
    )


def empty_notice(name: str) -> None:
    st.info(f"No data for {name} yet.")


# ------------------------------------------------------------------ headline

st.title("📊 Cooking Q&A Assistant — monitoring")

overview = db.fetch_one(
    """
    SELECT
        COUNT(*)            AS conversations,
        AVG(response_time)  AS avg_response_time,
        AVG(retrieval_time) AS avg_retrieval_time,
        SUM(total_tokens)   AS total_tokens,
        SUM(cost)           AS total_cost
    FROM conversations
    """
)

if not overview or not overview["conversations"]:
    st.warning(
        "No conversations logged yet. Ask something in the chat app first, "
        "then refresh this page."
    )
    st.stop()

thumbs = db.fetch_one(
    """
    SELECT
        COUNT(*) FILTER (WHERE score = 1)  AS up,
        COUNT(*) FILTER (WHERE score = -1) AS down
    FROM feedback
    WHERE source = 'user'
    """
) or {"up": 0, "down": 0}

rated = (thumbs["up"] or 0) + (thumbs["down"] or 0)
satisfaction = f"{(thumbs['up'] or 0) / rated:.0%}" if rated else "no ratings"

columns = st.columns(6)
columns[0].metric("Conversations", f"{overview['conversations']:,}")
columns[1].metric("Avg response", f"{overview['avg_response_time']:.2f}s")
columns[2].metric("Avg retrieval", f"{overview['avg_retrieval_time']:.2f}s")
columns[3].metric("Tokens used", f"{overview['total_tokens']:,}")
columns[4].metric("Estimated cost", f"${overview['total_cost']:.4f}")
columns[5].metric("Thumbs up", satisfaction, help=f"{rated} answers rated")

st.caption(
    f"Serving `{settings.RETRIEVAL_STRATEGY}` retrieval with the "
    f"`{settings.PROMPT_VARIANT}` prompt. Auto-refreshes every "
    f"{REFRESH_SECONDS}s; use R to force it."
)
st.divider()

# ------------------------------------------------------------ volume, timing

left, right = st.columns(2)

with left:
    st.subheader("1. Questions per hour")
    volume = frame(
        """
        SELECT date_trunc('hour', created_at) AS hour, COUNT(*) AS questions
        FROM conversations
        GROUP BY hour
        ORDER BY hour
        """
    )
    if volume.empty:
        empty_notice("question volume")
    else:
        st.altair_chart(
            bar(volume, "hour:T", "questions:Q", "hour", "questions"),
            use_container_width=True,
        )

with right:
    st.subheader("2. Response time over time")
    timings = frame(
        """
        SELECT
            date_trunc('hour', created_at) AS hour,
            AVG(response_time)  AS total,
            AVG(retrieval_time) AS retrieval
        FROM conversations
        GROUP BY hour
        ORDER BY hour
        """
    )
    if timings.empty:
        empty_notice("response times")
    else:
        melted = timings.melt("hour", var_name="phase", value_name="seconds")
        st.altair_chart(
            alt.Chart(melted)
            .mark_line(point=True)
            .encode(
                x=alt.X("hour:T", title="hour"),
                y=alt.Y("seconds:Q", title="seconds"),
                color=alt.Color("phase:N", title=None),
                tooltip=["hour:T", "phase:N", alt.Tooltip("seconds:Q", format=".2f")],
            )
            .properties(height=260),
            use_container_width=True,
        )

# --------------------------------------------------------- tokens, feedback

left, right = st.columns(2)

with left:
    st.subheader("3. Token usage per hour")
    tokens = frame(
        """
        SELECT
            date_trunc('hour', created_at) AS hour,
            SUM(prompt_tokens)     AS prompt,
            SUM(completion_tokens) AS completion
        FROM conversations
        GROUP BY hour
        ORDER BY hour
        """
    )
    if tokens.empty:
        empty_notice("token usage")
    else:
        melted = tokens.melt("hour", var_name="kind", value_name="tokens")
        st.altair_chart(
            alt.Chart(melted)
            .mark_area(opacity=0.8)
            .encode(
                x=alt.X("hour:T", title="hour"),
                y=alt.Y("tokens:Q", title="tokens", stack=True),
                color=alt.Color("kind:N", title=None),
                tooltip=["hour:T", "kind:N", "tokens:Q"],
            )
            .properties(height=260),
            use_container_width=True,
        )

with right:
    st.subheader("4. User feedback")
    feedback = frame(
        """
        SELECT
            CASE WHEN score = 1 THEN 'thumbs up' ELSE 'thumbs down' END AS rating,
            COUNT(*) AS count
        FROM feedback
        WHERE source = 'user' AND score IS NOT NULL
        GROUP BY rating
        """
    )
    if feedback.empty:
        empty_notice("user feedback")
    else:
        st.altair_chart(
            alt.Chart(feedback)
            .mark_bar(cornerRadiusEnd=2)
            .encode(
                x=alt.X("count:Q", title="answers"),
                y=alt.Y("rating:N", sort=["thumbs up", "thumbs down"], title=None),
                color=alt.Color(
                    "rating:N",
                    scale=alt.Scale(
                        domain=["thumbs up", "thumbs down"],
                        range=["#2e7d32", "#c62828"],
                    ),
                    legend=None,
                ),
                tooltip=["rating", "count"],
            )
            .properties(height=200),
            use_container_width=True,
        )

# ----------------------------------------------------------- judge verdicts

left, right = st.columns(2)

with left:
    st.subheader("5. Judge: relevance")
    relevance = frame(
        """
        SELECT relevance, COUNT(*) AS count
        FROM feedback
        WHERE source = 'judge' AND relevance IS NOT NULL
        GROUP BY relevance
        """
    )
    if relevance.empty:
        empty_notice("judge relevance")
        st.caption("Enable 'Score answers with the LLM judge' in the chat sidebar.")
    else:
        st.altair_chart(
            ordered_bar(relevance, "relevance", RELEVANCE_ORDER,
                        "Does the answer address the question?"),
            use_container_width=True,
        )

with right:
    st.subheader("6. Judge: groundedness")
    groundedness = frame(
        """
        SELECT groundedness, COUNT(*) AS count
        FROM feedback
        WHERE source = 'judge' AND groundedness IS NOT NULL
        GROUP BY groundedness
        """
    )
    if groundedness.empty:
        empty_notice("judge groundedness")
        st.caption("Same switch as the chart on the left.")
    else:
        st.altair_chart(
            ordered_bar(groundedness, "groundedness", GROUNDEDNESS_ORDER,
                        "Is every claim backed by the retrieved context?"),
            use_container_width=True,
        )

# ------------------------------------------------------------ topics, spread

left, right = st.columns(2)

with left:
    st.subheader("7. Most asked-about topics")
    topics = frame(
        """
        SELECT top_tag AS topic, COUNT(*) AS questions
        FROM conversations
        WHERE top_tag IS NOT NULL
        GROUP BY topic
        ORDER BY questions DESC
        LIMIT 12
        """
    )
    if topics.empty:
        empty_notice("topics")
    else:
        st.altair_chart(
            alt.Chart(topics)
            .mark_bar(color="#54a24b", cornerRadiusEnd=2)
            .encode(
                x=alt.X("questions:Q", title="questions"),
                y=alt.Y("topic:N", sort="-x", title=None),
                tooltip=["topic", "questions"],
            )
            .properties(height=300),
            use_container_width=True,
        )

with right:
    st.subheader("8. Response time distribution")
    spread = frame("SELECT response_time FROM conversations")
    if spread.empty:
        empty_notice("response times")
    else:
        st.altair_chart(
            alt.Chart(spread)
            .mark_bar(color="#e45756")
            .encode(
                x=alt.X("response_time:Q", bin=alt.Bin(maxbins=20),
                        title="seconds"),
                y=alt.Y("count():Q", title="answers"),
                tooltip=[alt.Tooltip("count():Q", title="answers")],
            )
            .properties(height=300),
            use_container_width=True,
        )

# ----------------------------------------------------- per-strategy summary

st.divider()
st.subheader("Per-strategy summary")

by_strategy = frame(
    """
    SELECT
        c.strategy,
        c.prompt_variant,
        c.model,
        COUNT(*)                                            AS answers,
        ROUND(AVG(c.response_time)::numeric, 2)             AS avg_seconds,
        ROUND(AVG(c.total_tokens)::numeric, 0)              AS avg_tokens,
        COUNT(f.id) FILTER (WHERE f.score = 1)              AS thumbs_up,
        COUNT(f.id) FILTER (WHERE f.score = -1)             AS thumbs_down
    FROM conversations AS c
    LEFT JOIN feedback AS f
        ON f.conversation_id = c.id AND f.source = 'user'
    GROUP BY c.strategy, c.prompt_variant, c.model
    ORDER BY answers DESC
    """
)
if by_strategy.empty:
    empty_notice("strategy comparison")
else:
    st.dataframe(by_strategy, use_container_width=True, hide_index=True)

# -------------------------------------------------------------- recent rows

st.subheader("Recent questions")

recent = frame(
    """
    SELECT
        c.created_at,
        c.question,
        c.top_tag,
        c.strategy,
        ROUND(c.response_time::numeric, 2) AS seconds,
        c.total_tokens,
        MAX(f.score) FILTER (WHERE f.source = 'user')      AS user_score,
        MAX(f.relevance) FILTER (WHERE f.source = 'judge') AS judge_relevance
    FROM conversations AS c
    LEFT JOIN feedback AS f ON f.conversation_id = c.id
    GROUP BY c.id, c.created_at, c.question, c.top_tag, c.strategy,
             c.response_time, c.total_tokens
    ORDER BY c.created_at DESC
    LIMIT 25
    """
)
if recent.empty:
    empty_notice("recent questions")
else:
    st.dataframe(recent, use_container_width=True, hide_index=True)
