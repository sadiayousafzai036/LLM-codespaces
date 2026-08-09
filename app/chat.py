"""Streamlit chat interface.

    uv run streamlit run app/chat.py

Two things about Streamlit shaped this file.

First, the whole script re-runs on every interaction, so anything expensive has
to be cached. Embedding the corpus takes tens of seconds, and the retriever
factory is cached once and shared by every strategy so switching strategies in
the sidebar does not re-embed anything.

Second, a button click triggers a re-run in which the button is no longer
pressed. Answering inside an `if st.button(...)` block and rendering there works
until the first feedback click wipes the answer off the screen. So the result
lives in session state and rendering always reads from there.
"""

import streamlit as st

from app import db, judge, settings
from app.assistant import load_knowledge_base
from app.llm import LLMClient
from app.rag import PROMPT_VARIANTS, CookingAssistant, build_context
from app.retrieval import STRATEGIES, RetrieverFactory

st.set_page_config(page_title="Cooking Q&A Assistant", page_icon="🍳")

EXAMPLES = [
    "why did my bread turn out dense and gummy?",
    "can I leave cooked rice out overnight?",
    "how do I stop my custard from splitting?",
    "what is the difference between baking soda and baking powder?",
]


@st.cache_resource(show_spinner="Loading the knowledge base and embedding it...")
def get_factory() -> RetrieverFactory:
    documents = load_knowledge_base()
    return RetrieverFactory(documents, llm=LLMClient(), show_progress=False)


@st.cache_resource(show_spinner="Preparing retriever...")
def get_assistant(strategy: str, prompt_variant: str, model: str) -> CookingAssistant:
    factory = get_factory()
    return CookingAssistant(
        retriever=factory.build(strategy),
        llm=factory.llm,
        prompt_variant=prompt_variant,
        model=model,
    )


@st.cache_data(ttl=600)
def get_tags() -> list[str]:
    return db.list_tags()


@st.cache_data(ttl=600)
def get_document_count() -> int:
    return db.count_documents()


def render_sidebar() -> dict:
    with st.sidebar:
        st.header("Settings")

        # A typo in RETRIEVAL_STRATEGY should not take the whole app down, so
        # fall back to the first strategy and say so.
        if settings.RETRIEVAL_STRATEGY in STRATEGIES:
            default_strategy = STRATEGIES.index(settings.RETRIEVAL_STRATEGY)
        else:
            default_strategy = 0
            st.warning(
                f"RETRIEVAL_STRATEGY='{settings.RETRIEVAL_STRATEGY}' is not a "
                f"known strategy, defaulting to '{STRATEGIES[0]}'."
            )

        strategy = st.selectbox(
            "Retrieval strategy",
            STRATEGIES,
            index=default_strategy,
            help="The default is whichever strategy won the retrieval evaluation.",
        )
        variants = sorted(PROMPT_VARIANTS)
        prompt_variant = st.selectbox(
            "Prompt variant",
            variants,
            index=variants.index(settings.PROMPT_VARIANT)
            if settings.PROMPT_VARIANT in variants
            else 0,
            help="Compared in evaluation/rag_eval.py.",
        )
        model = st.text_input("Model", value=settings.LLM_MODEL)

        tags = get_tags()
        tag = st.selectbox(
            "Limit to tag",
            ["(all topics)"] + tags,
            help="Narrows the search to one topic before ranking.",
        )

        judge_answers = st.checkbox(
            "Score answers with the LLM judge",
            value=False,
            help=(
                "Adds one extra LLM call per question and records the verdict "
                "on the dashboard. Off by default to stay inside free-tier limits."
            ),
        )

        st.divider()
        st.caption(f"{get_document_count():,} documents indexed")
        st.caption(f"Source: {settings.SE_SITE}.stackexchange.com, CC BY-SA")

    return {
        "strategy": strategy,
        "prompt_variant": prompt_variant,
        "model": model.strip() or settings.LLM_MODEL,
        "tag": None if tag == "(all topics)" else tag,
        "judge_answers": judge_answers,
    }


def ask(question: str, options: dict) -> None:
    assistant = get_assistant(
        options["strategy"], options["prompt_variant"], options["model"]
    )
    with st.spinner("Searching the Q&A and writing an answer..."):
        result = assistant.answer(question, tag=options["tag"])

    conversation_id = db.save_conversation(result.as_conversation_row())

    verdict = None
    if options["judge_answers"] and result.documents:
        with st.spinner("Scoring the answer..."):
            try:
                verdict, _ = judge.evaluate(
                    question, result.answer, build_context(result.documents)
                )
                db.save_feedback(
                    conversation_id,
                    source="judge",
                    relevance=verdict.relevance,
                    groundedness=verdict.groundedness,
                    explanation=verdict.explanation,
                )
            except RuntimeError as exc:
                st.warning(f"The judge did not return a verdict: {exc}")

    st.session_state.result = result
    st.session_state.conversation_id = conversation_id
    st.session_state.verdict = verdict


def _md_escape(text: str) -> str:
    """Square brackets in a title would break the markdown link around it.

    Stack Exchange titles do contain them, e.g. "Why is my roux grainy? [UK]".
    """
    return text.replace("[", "\\[").replace("]", "\\]")


def render_result() -> None:
    result = st.session_state.get("result")
    if result is None:
        return

    conversation_id = st.session_state.conversation_id

    st.markdown(f"**You asked:** {result.question}")
    st.markdown(result.answer)

    if result.rewritten_query:
        st.caption(f"Searched for: _{result.rewritten_query}_")

    verdict = st.session_state.get("verdict")
    if verdict is not None:
        st.info(
            f"Judge: relevance **{verdict.relevance}**, "
            f"groundedness **{verdict.groundedness}** — {verdict.explanation}"
        )

    if result.documents:
        with st.expander(f"Sources ({len(result.documents)})"):
            for number, document in enumerate(result.documents, start=1):
                accepted = " · accepted answer" if document["is_accepted"] else ""
                title = _md_escape(document["title"])
                link = document["link"]
                heading = (
                    f"**[{number}] [{title}]({link})**"
                    if link
                    else f"**[{number}] {title}**"
                )
                st.markdown(
                    f"{heading}  \n"
                    f"`{document['primary_tag']}` · "
                    f"score {document['answer_score']}{accepted}"
                )
                st.caption(document["attribution"])

    left, middle, right = st.columns(3)
    left.metric("Retrieval", f"{result.retrieval_time:.2f}s")
    middle.metric("Total", f"{result.response_time:.2f}s")
    right.metric("Tokens", f"{result.usage.total_tokens:,}")
    st.caption(
        f"strategy `{result.strategy}` · prompt `{result.prompt_variant}` · "
        f"model `{result.model}`"
    )

    render_feedback(conversation_id)


def render_feedback(conversation_id: int) -> None:
    """Thumbs, rendered from what is already stored rather than from click state.

    Reading the saved score back means the choice survives the re-run that the
    click itself causes, and a second click just overwrites the first.
    """
    existing = db.user_feedback_for(conversation_id)

    st.write("Was this answer useful?")
    up, down, _ = st.columns([1, 1, 6])

    if up.button("👍", key=f"up_{conversation_id}",
                 type="primary" if existing == 1 else "secondary"):
        db.save_feedback(conversation_id, source="user", score=1)
        st.rerun()

    if down.button("👎", key=f"down_{conversation_id}",
                   type="primary" if existing == -1 else "secondary"):
        db.save_feedback(conversation_id, source="user", score=-1)
        st.rerun()

    if existing is not None:
        st.caption("Thanks, that is recorded on the dashboard.")


def preflight() -> bool:
    """Turn the two likely first-run failures into readable messages.

    Without this, a missing API key or an un-ingested database surfaces as a
    Streamlit traceback, which is a poor first impression and buries the fix.
    """
    if not settings.GROQ_API_KEY:
        st.error(
            "**GROQ_API_KEY is not set.** Copy `.env.example` to `.env` and add "
            "a key from https://console.groq.com/keys, then restart the app."
        )
        return False

    try:
        if get_document_count() == 0:
            st.error(
                "**The knowledge base is empty.** Run `make ingest` "
                "(or `docker compose up ingest`) to load the cooking Q&A, "
                "then refresh."
            )
            return False
    except Exception as exc:
        st.error(
            f"**Cannot reach Postgres.** Start it with `make db`, then refresh.\n\n"
            f"```\n{exc}\n```"
        )
        return False

    return True


def main() -> None:
    st.title("🍳 Cooking Q&A Assistant")
    st.caption(
        "Answers cooking questions from the highest-voted question and answer "
        "pairs on the Seasoned Advice Stack Exchange, with the sources it used."
    )

    if not preflight():
        st.stop()

    options = render_sidebar()

    with st.form("ask"):
        question = st.text_area(
            "Your cooking question",
            placeholder=EXAMPLES[0],
            height=80,
        )
        submitted = st.form_submit_button("Ask", type="primary")

    st.caption("Try: " + " · ".join(f"_{example}_" for example in EXAMPLES[1:]))

    if submitted:
        if question.strip():
            ask(question.strip(), options)
        else:
            st.warning("Type a question first.")

    render_result()


main()
