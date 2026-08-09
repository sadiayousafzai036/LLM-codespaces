"""Generate the ground truth question set used by both evaluations.

For a sample of documents we ask the LLM for questions a home cook might type
that this particular answer would resolve. The document that produced a question
is that question's one relevant result, which is what makes hit rate and MRR
computable without hand-labelling hundreds of pairs.

The obvious shortcut would be to use each question's own Stack Exchange title as
the query. That inflates the numbers badly: titles are indexed verbatim, so
keyword search scores near-perfectly and the comparison between strategies
becomes meaningless. Generated paraphrases keep the task honest.

    uv run python -m evaluation.ground_truth --documents 120 --per-document 3
"""

import argparse
import random

from pydantic import BaseModel, Field
from tqdm import tqdm

from app import db, settings
from app.llm import LLMClient
from evaluation.common import GROUND_TRUTH_CSV, RANDOM_SEED, write_csv

# Sampling only from answers the community endorsed keeps the generated
# questions answerable. A question paraphrased from a poor answer is unfair to
# every strategy equally, but it adds noise for no benefit.
MIN_ANSWER_SCORE = 5


class GeneratedQuestions(BaseModel):
    questions: list[str] = Field(description="Distinct questions, one per string")


INSTRUCTIONS = """
You write realistic search queries for a cooking Q&A site.

You are given one question-and-answer pair. Write {count} different questions
that a home cook might type, each of which this answer would resolve.

Rules:
- Do not reuse the wording of the original title. Rephrase completely.
- Write the way people actually type: "why is my custard grainy", not "What are
  the causes of graininess in custard preparations?".
- Vary the form across the {count} questions: a symptom, a how-to, a why.
- Each question must be answerable from this answer alone. Do not invent
  details the answer does not discuss.
- One sentence each, no numbering, no explanation.
""".strip()

PROMPT = """
TITLE: {title}

QUESTION: {question}

ANSWER: {answer}
""".strip()


def sample_documents(documents: list[dict], count: int) -> list[dict]:
    """Spread the sample across tags so one popular topic cannot dominate.

    Round-robin over tags rather than a flat random sample: the corpus is
    heavily skewed toward a few tags, and a flat sample would mostly measure
    performance on those.
    """
    eligible = [
        document
        for document in documents
        if document["is_accepted"] or document["answer_score"] >= MIN_ANSWER_SCORE
    ]
    if not eligible:
        raise RuntimeError("no documents passed the quality filter")

    rng = random.Random(RANDOM_SEED)

    by_tag: dict[str, list[dict]] = {}
    for document in eligible:
        by_tag.setdefault(document["primary_tag"], []).append(document)
    for bucket in by_tag.values():
        rng.shuffle(bucket)

    tags = sorted(by_tag)
    rng.shuffle(tags)

    selected: list[dict] = []
    while len(selected) < count:
        added_this_round = 0
        for tag in tags:
            if by_tag[tag]:
                selected.append(by_tag[tag].pop())
                added_this_round += 1
                if len(selected) >= count:
                    break
        if added_this_round == 0:
            print(f"only {len(selected)} documents were eligible, wanted {count}")
            break
    return selected


def generate(documents: int, per_document: int, model: str | None) -> list[dict]:
    knowledge_base = db.load_documents()
    if not knowledge_base:
        raise RuntimeError("the documents table is empty - run `make ingest` first")

    sample = sample_documents(knowledge_base, documents)
    print(
        f"generating {per_document} questions for each of {len(sample)} documents "
        f"across {len({d['primary_tag'] for d in sample})} tags"
    )

    llm = LLMClient()
    instructions = INSTRUCTIONS.format(count=per_document)

    rows: list[dict] = []
    failures = 0

    for document in tqdm(sample, desc="documents"):
        prompt = PROMPT.format(
            title=document["title"],
            question=document["question_text"][:800],
            answer=document["answer_text"][:1500],
        )
        try:
            generated, _ = llm.structured(
                instructions, prompt, GeneratedQuestions, model=model
            )
        except RuntimeError as exc:
            failures += 1
            print(f"  skipped {document['doc_id']}: {exc}")
            continue

        for question in generated.questions[:per_document]:
            question = question.strip()
            if question:
                rows.append(
                    {
                        "doc_id": document["doc_id"],
                        "primary_tag": document["primary_tag"],
                        "question": question,
                    }
                )

    if failures:
        print(f"{failures} documents produced no questions")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=120,
                        help="how many documents to sample")
    parser.add_argument("--per-document", type=int, default=3,
                        help="questions to generate per document")
    parser.add_argument("--model", default=settings.LLM_MODEL)
    args = parser.parse_args()

    rows = generate(args.documents, args.per_document, args.model)
    if not rows:
        raise RuntimeError("no questions were generated")

    write_csv(GROUND_TRUTH_CSV, rows, ["doc_id", "primary_tag", "question"])
    print(f"{len(rows)} questions over {len({r['doc_id'] for r in rows})} documents")


if __name__ == "__main__":
    main()
