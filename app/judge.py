"""LLM-as-a-judge scoring for generated answers.

Two verdicts come back from a single call:

  relevance    does the answer address the question the user actually asked
  groundedness is every claim in the answer supported by the retrieved context

Relevance alone is not enough for a RAG system. An answer can be perfectly
on-topic and still be invented, which is the failure mode that matters when the
subject is food safety. Asking for both in one call also halves the number of
requests, which is the binding constraint on a free API tier.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app import settings
from app.llm import LLMClient, Usage

RELEVANCE_LEVELS = ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]
GROUNDEDNESS_LEVELS = ["SUPPORTED", "PARTLY_SUPPORTED", "UNSUPPORTED"]


class Verdict(BaseModel):
    relevance: Literal["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]
    groundedness: Literal["SUPPORTED", "PARTLY_SUPPORTED", "UNSUPPORTED"]
    explanation: str = Field(description="One or two sentences, no more")


JUDGE_INSTRUCTIONS = """
You evaluate answers produced by a cooking question-answering system.

Score two things independently.

relevance - does the answer address the question that was asked?
  RELEVANT        it answers the question
  PARTLY_RELEVANT it addresses part of the question, or answers a near-miss
  NON_RELEVANT    it does not address the question

groundedness - is every factual claim supported by the context provided?
  SUPPORTED         every claim traces back to the context
  PARTLY_SUPPORTED  the main claims are supported, some details are not
  UNSUPPORTED       key claims appear nowhere in the context

Judge groundedness against the context only. A claim that is true in general but
absent from the context is not supported. An honest "the context does not cover
this" is SUPPORTED, and is RELEVANT when that is genuinely the case.

Be strict. If you are unsure, choose the lower level.
""".strip()

JUDGE_PROMPT = """
QUESTION:
{question}

CONTEXT GIVEN TO THE SYSTEM:
{context}

ANSWER PRODUCED:
{answer}
""".strip()

# The judge only needs enough context to check the claims, and judging runs over
# the whole evaluation set, so we trim to keep inside the free-tier rate limit.
MAX_CONTEXT_CHARS = 6000


def evaluate(question: str, answer: str, context: str,
             llm: LLMClient | None = None,
             model: str | None = None) -> tuple[Verdict, Usage]:
    llm = llm or LLMClient()
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context[:MAX_CONTEXT_CHARS],
        answer=answer,
    )
    return llm.structured(
        JUDGE_INSTRUCTIONS, prompt, Verdict, model=model or settings.JUDGE_MODEL
    )
