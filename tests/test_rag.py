"""Tests for prompt assembly and the result record.

The context block is what the model actually sees, and the conversation row is
what the dashboard actually charts. Both are easy to break silently.
"""

from app.rag import (
    NO_CONTEXT_ANSWER,
    PROMPT_VARIANTS,
    CookingAssistant,
    RagResult,
    build_context,
)
from app.retrieval import Retrieval

import pytest


def document(doc_id, tag="baking", accepted=False, title=None):
    return {
        "doc_id": doc_id,
        "question_id": int(doc_id.lstrip("d")),
        "title": title or f"Title {doc_id}",
        "question_text": f"Question body {doc_id}",
        "answer_text": f"Answer body {doc_id}",
        "tags": tag,
        "primary_tag": tag,
        "question_score": 10,
        "answer_score": 5,
        "is_accepted": accepted,
        "link": f"https://cooking.stackexchange.com/q/{doc_id}",
        "attribution": "Answer by someone",
    }


class StubRetriever:
    name = "stub"

    def __init__(self, documents):
        self._documents = documents

    def retrieve(self, query, num_results=5, tag=None):
        return Retrieval(documents=self._documents[:num_results])


class StubLLM:
    def __init__(self):
        self.calls = []

    def complete(self, instructions, prompt, model=None):
        self.calls.append((instructions, prompt, model))
        from app.llm import EMPTY_USAGE, LLMResult

        return LLMResult(text="an answer", model=model or "stub",
                         usage=EMPTY_USAGE, latency=0.0)


# ------------------------------------------------------------- build_context


def test_context_numbers_sources_from_one():
    context = build_context([document("d1"), document("d2")])
    assert context.startswith("[1] ")
    assert "[2] " in context


def test_context_marks_the_accepted_answer():
    context = build_context([document("d1", accepted=True)])
    assert "(accepted answer)" in context


def test_context_does_not_mark_unaccepted_answers():
    assert "(accepted answer)" not in build_context([document("d1")])


def test_context_includes_title_question_answer_and_link():
    context = build_context([document("d1")])
    for expected in ("Title d1", "Question body d1", "Answer body d1",
                     "cooking.stackexchange.com"):
        assert expected in context


def test_context_of_no_documents_is_empty():
    assert build_context([]) == ""


# ----------------------------------------------------------------- RagResult


def test_doc_ids_preserve_retrieval_order():
    result = RagResult(question="q", answer="a",
                       documents=[document("d3"), document("d1")])
    assert result.doc_ids == ["d3", "d1"]


def test_top_tag_picks_the_majority():
    result = RagResult(
        question="q", answer="a",
        documents=[document("d1", "baking"), document("d2", "baking"),
                   document("d3", "food-safety")],
    )
    assert result.top_tag == "baking"


def test_top_tag_of_no_documents_is_none():
    assert RagResult(question="q", answer="a").top_tag is None


def test_conversation_row_flattens_doc_ids_to_a_string():
    result = RagResult(question="q", answer="a",
                       documents=[document("d1"), document("d2")])
    assert result.as_conversation_row()["retrieved_doc_ids"] == "d1,d2"


def test_conversation_row_has_every_column_db_expects():
    row = RagResult(question="q", answer="a").as_conversation_row()
    expected = {
        "question", "answer", "tag_filter", "strategy", "model",
        "prompt_variant", "rewritten_query", "retrieved_doc_ids", "top_tag",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "retrieval_time", "response_time", "cost",
    }
    assert set(row) == expected


# ------------------------------------------------------------ the flow itself


def test_empty_retrieval_skips_the_llm_entirely():
    llm = StubLLM()
    assistant = CookingAssistant(StubRetriever([]), llm, prompt_variant="grounded")

    result = assistant.answer("anything")

    assert result.answer == NO_CONTEXT_ANSWER
    assert llm.calls == []


def test_empty_retrieval_still_reports_a_total_time():
    assistant = CookingAssistant(StubRetriever([]), StubLLM(),
                                 prompt_variant="grounded")
    result = assistant.answer("anything")
    assert result.response_time >= result.retrieval_time


def test_answer_uses_the_requested_prompt_variant():
    llm = StubLLM()
    assistant = CookingAssistant(StubRetriever([document("d1")]), llm,
                                 prompt_variant="concise")

    assistant.answer("why is my bread dense?")

    assert llm.calls[0][0] == PROMPT_VARIANTS["concise"]


def test_answer_puts_the_question_in_the_prompt():
    llm = StubLLM()
    assistant = CookingAssistant(StubRetriever([document("d1")]), llm,
                                 prompt_variant="grounded")

    assistant.answer("why is my bread dense?")

    assert "why is my bread dense?" in llm.calls[0][1]


def test_unknown_prompt_variant_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown prompt variant"):
        CookingAssistant(StubRetriever([]), StubLLM(), prompt_variant="nonsense")


def test_every_variant_tells_the_model_it_is_a_cooking_assistant():
    for name, instructions in PROMPT_VARIANTS.items():
        assert "cooking" in instructions.lower(), name
