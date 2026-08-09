"""The RAG flow: retrieve, build a prompt, answer, and record what happened.

Three prompt variants are compared in evaluation/rag_eval.py:

  concise     minimal instructions, closest to a naive baseline
  grounded    refuses to go beyond the context and cites the sources it used
  structured  grounded, plus a fixed short answer / details / caveats shape

`PROMPT_VARIANT` in the environment decides which one the app serves.
"""

import time
from dataclasses import dataclass, field

from app import settings
from app.llm import EMPTY_USAGE, LLMClient, Usage
from app.retrieval import Retrieval

# ------------------------------------------------------------------- prompts

CONCISE = """
You answer cooking questions using the provided context.
Keep the answer short.
""".strip()

GROUNDED = """
You are a cooking assistant. Answer the user's question using only the numbered
context below, which comes from a cooking Q&A site.

Rules:
- Use only what the context says. Do not add techniques, times or temperatures
  from your own knowledge, even when you are confident they are right.
- If the context does not answer the question, say so plainly and describe what
  it does cover. Do not guess.
- When sources disagree, say they disagree and give both positions.
- Cite the sources you used by their numbers, like [2], next to the claim.
- Write for a home cook: plain language, no lecturing, no recipe preamble.
""".strip()

STRUCTURED = (
    GROUNDED
    + """

Format the answer as:

**Short answer:** one or two sentences.

**Details:** the reasoning or method, as prose or a short list.

**Worth knowing:** caveats, food-safety notes or disagreements between sources.
Leave this section out if the context raises none.
"""
)

PROMPT_VARIANTS = {
    "concise": CONCISE,
    "grounded": GROUNDED,
    "structured": STRUCTURED,
}

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


def build_context(documents: list[dict]) -> str:
    """Number the sources so the model has something concrete to cite.

    Each entry carries the original question title as well as the answer,
    because an answer taken alone often reads as advice with no subject.
    """
    blocks = []
    for number, document in enumerate(documents, start=1):
        accepted = " (accepted answer)" if document.get("is_accepted") else ""
        blocks.append(
            f"[{number}] {document['title']}{accepted}\n"
            f"Asked: {document['question_text']}\n"
            f"Answer: {document['answer_text']}\n"
            f"Source: {document['link']}"
        )
    return "\n\n".join(blocks)


# ------------------------------------------------------------------- results


@dataclass
class RagResult:
    question: str
    answer: str
    documents: list[dict] = field(default_factory=list)
    strategy: str = ""
    prompt_variant: str = ""
    model: str = ""
    tag_filter: str | None = None
    rewritten_query: str | None = None
    prompt: str = ""
    usage: Usage = EMPTY_USAGE
    retrieval_time: float = 0.0
    response_time: float = 0.0

    @property
    def doc_ids(self) -> list[str]:
        return [document["doc_id"] for document in self.documents]

    @property
    def top_tag(self) -> str | None:
        """Most common tag among the retrieved documents, for the dashboard."""
        if not self.documents:
            return None
        tags = [document["primary_tag"] for document in self.documents]
        return max(set(tags), key=tags.count)

    def as_conversation_row(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "tag_filter": self.tag_filter,
            "strategy": self.strategy,
            "model": self.model,
            "prompt_variant": self.prompt_variant,
            "rewritten_query": self.rewritten_query,
            # Stored as a plain string: it is only ever read back for display,
            # so an array column would buy nothing.
            "retrieved_doc_ids": ",".join(self.doc_ids),
            "top_tag": self.top_tag,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
            "retrieval_time": self.retrieval_time,
            "response_time": self.response_time,
            "cost": self.usage.cost,
        }


NO_CONTEXT_ANSWER = (
    "I could not find anything about that in the cooking Q&A I have indexed, "
    "so I would rather not guess. Try rephrasing the question, or clear the tag "
    "filter if you set one."
)


class CookingAssistant:
    def __init__(self, retriever, llm: LLMClient,
                 prompt_variant: str | None = None, model: str | None = None,
                 num_results: int | None = None):
        variant = prompt_variant or settings.PROMPT_VARIANT
        if variant not in PROMPT_VARIANTS:
            raise ValueError(
                f"unknown prompt variant '{variant}', "
                f"expected one of {sorted(PROMPT_VARIANTS)}"
            )

        self.retriever = retriever
        self.llm = llm
        self.prompt_variant = variant
        self.instructions = PROMPT_VARIANTS[variant]
        self.model = model or settings.LLM_MODEL
        self.num_results = num_results or settings.NUM_RESULTS

    def answer(self, question: str, tag: str | None = None) -> RagResult:
        started = time.monotonic()
        retrieval: Retrieval = self.retriever.retrieve(
            question, num_results=self.num_results, tag=tag
        )
        retrieval_time = time.monotonic() - started

        base = RagResult(
            question=question,
            answer="",
            documents=retrieval.documents,
            strategy=self.retriever.name,
            prompt_variant=self.prompt_variant,
            model=self.model,
            tag_filter=tag,
            rewritten_query=retrieval.rewritten_query,
            usage=retrieval.usage,
            retrieval_time=retrieval_time,
        )

        # Retrieval genuinely returning nothing is different from retrieval
        # returning something irrelevant. Spending a call to have the model
        # say "no context" adds latency and cost for a fixed answer.
        if not retrieval.documents:
            base.answer = NO_CONTEXT_ANSWER
            # Without this the dashboard shows a row with real retrieval time and
            # zero total time, which reads as a broken measurement.
            base.response_time = retrieval_time
            return base

        prompt = PROMPT_TEMPLATE.format(
            question=question, context=build_context(retrieval.documents)
        )
        result = self.llm.complete(self.instructions, prompt, model=self.model)

        base.answer = result.text
        base.prompt = prompt
        base.usage = retrieval.usage + result.usage
        base.response_time = time.monotonic() - started
        return base
