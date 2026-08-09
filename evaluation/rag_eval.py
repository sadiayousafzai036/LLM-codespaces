"""Compare prompt variants and models with an LLM judge.

Run after `make eval-ground-truth`:

    uv run python -m evaluation.rag_eval --questions 20

Every combination of three prompt variants and two models answers the same
sample of questions, and a judge scores each answer for relevance and
groundedness.

Retrieval is held fixed at `hybrid` throughout, even though the app serves
`hybrid_rerank`. Two reasons: the point here is to isolate the effect of the
prompt and the model, and a re-ranking step would add one LLM call per answer to
a script that already makes two.

Cost warning: the default 20 questions means 20 x 3 x 2 = 120 answers plus 120
judgements. On a free tier, expect this to take a while and to hit rate limits
that the client backs off from. Lower --questions if you just want a smoke test.
"""

import argparse
import random

from tqdm import tqdm

from app import db, judge, settings
from app.llm import LLMClient
from app.rag import PROMPT_VARIANTS, CookingAssistant, build_context
from app.retrieval import RetrieverFactory
from evaluation.common import (
    GROUND_TRUTH_CSV,
    RAG_ANSWERS_CSV,
    RAG_CSV,
    RANDOM_SEED,
    markdown_table,
    read_csv,
    write_csv,
)

# Groundedness is weighted equally with relevance. An answer that is on-topic but
# invented is worse than useless for a question about food safety, so a variant
# cannot win this comparison on relevance alone.
RELEVANCE_WEIGHT = 0.5
GROUNDEDNESS_WEIGHT = 0.5


def sample_questions(ground_truth: list[dict], count: int) -> list[dict]:
    """One question per source document, so the sample covers distinct topics."""
    seen: set[str] = set()
    unique = []
    for row in ground_truth:
        if row["doc_id"] not in seen:
            seen.add(row["doc_id"])
            unique.append(row)

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(unique)
    return unique[:count]


def run_combination(assistant: CookingAssistant, questions: list[dict],
                    llm: LLMClient, judge_model: str, label: str) -> list[dict]:
    rows = []
    for row in tqdm(questions, desc=label.ljust(28), leave=False):
        result = assistant.answer(row["question"])
        context = build_context(result.documents)

        try:
            verdict, judge_usage = judge.evaluate(
                row["question"], result.answer, context,
                llm=llm, model=judge_model,
            )
            relevance = verdict.relevance
            groundedness = verdict.groundedness
            explanation = verdict.explanation
        except RuntimeError as exc:
            # A judge that will not answer must not silently count as a pass.
            print(f"  judge failed on {row['doc_id']}: {exc}")
            relevance = groundedness = "JUDGE_FAILED"
            explanation = str(exc)
            judge_usage = None

        rows.append(
            {
                "prompt_variant": assistant.prompt_variant,
                "model": assistant.model,
                "doc_id": row["doc_id"],
                "question": row["question"],
                "answer": result.answer,
                "relevance": relevance,
                "groundedness": groundedness,
                "explanation": explanation,
                "answer_tokens": result.usage.total_tokens,
                "judge_tokens": judge_usage.total_tokens if judge_usage else 0,
                "response_time": round(result.response_time, 3),
                "cost": round(result.usage.cost, 6),
            }
        )
    return rows


def summarise(rows: list[dict]) -> dict:
    total = len(rows)
    judged = [row for row in rows if row["relevance"] != "JUDGE_FAILED"]
    denominator = len(judged) or 1

    relevant = sum(1 for row in judged if row["relevance"] == "RELEVANT")
    partly = sum(1 for row in judged if row["relevance"] == "PARTLY_RELEVANT")
    supported = sum(1 for row in judged if row["groundedness"] == "SUPPORTED")

    relevant_rate = relevant / denominator
    supported_rate = supported / denominator

    return {
        "prompt_variant": rows[0]["prompt_variant"],
        "model": rows[0]["model"],
        "answers": total,
        "judged": len(judged),
        "relevant": relevant_rate,
        "partly_relevant": partly / denominator,
        "supported": supported_rate,
        "score": RELEVANCE_WEIGHT * relevant_rate
        + GROUNDEDNESS_WEIGHT * supported_rate,
        "avg_tokens": sum(row["answer_tokens"] for row in rows) / total,
        "avg_seconds": sum(row["response_time"] for row in rows) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument("--retrieval", default="hybrid",
                        help="retrieval strategy held fixed across combinations")
    parser.add_argument("--variants", nargs="+", default=sorted(PROMPT_VARIANTS))
    parser.add_argument("--models", nargs="+",
                        default=[settings.LLM_MODEL, settings.LLM_MODEL_FAST])
    parser.add_argument("--judge-model", default=settings.JUDGE_MODEL)
    args = parser.parse_args()

    questions = sample_questions(read_csv(GROUND_TRUTH_CSV), args.questions)
    if not questions:
        raise RuntimeError(
            "no questions to evaluate - check the ground truth file and that "
            "--questions is above zero"
        )
    documents = db.load_documents()
    if not documents:
        raise RuntimeError("the documents table is empty - run `make ingest` first")

    llm = LLMClient()
    factory = RetrieverFactory(documents, llm=llm)
    retriever = factory.build(args.retrieval)

    models = list(dict.fromkeys(args.models))
    combinations = [(v, m) for m in models for v in args.variants]
    print(
        f"{len(questions)} questions x {len(combinations)} combinations "
        f"= {len(questions) * len(combinations)} answers, each one judged"
    )
    print(f"retrieval held at '{args.retrieval}', judge is '{args.judge_model}'")

    all_answers: list[dict] = []
    summaries: list[dict] = []

    for variant, model in combinations:
        assistant = CookingAssistant(
            retriever=retriever, llm=llm, prompt_variant=variant, model=model
        )
        rows = run_combination(
            assistant, questions, llm, args.judge_model, f"{variant} / {model}"
        )
        all_answers.extend(rows)
        summary = summarise(rows)
        summaries.append(summary)
        print(
            f"  {variant:<11} {model:<28} "
            f"relevant {summary['relevant']:.2f}  "
            f"supported {summary['supported']:.2f}  "
            f"score {summary['score']:.3f}"
        )

    write_csv(RAG_ANSWERS_CSV, all_answers)

    columns = [
        "prompt_variant", "model", "answers", "judged", "relevant",
        "partly_relevant", "supported", "score", "avg_tokens", "avg_seconds",
    ]
    write_csv(RAG_CSV, summaries, columns)

    print()
    print(markdown_table(
        sorted(summaries, key=lambda row: row["score"], reverse=True),
        columns,
        {
            "relevant": ".2f", "partly_relevant": ".2f", "supported": ".2f",
            "score": ".3f", "avg_tokens": ".0f", "avg_seconds": ".2f",
        },
    ))

    best = max(summaries, key=lambda row: row["score"])
    print()
    print(f"best combination: {best['prompt_variant']} on {best['model']}")
    print(
        f"set PROMPT_VARIANT={best['prompt_variant']} and "
        f"LLM_MODEL={best['model']} in .env to serve it"
    )


if __name__ == "__main__":
    main()
