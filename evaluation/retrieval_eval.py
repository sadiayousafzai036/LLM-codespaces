"""Compare retrieval strategies on hit rate and MRR.

Run after `make eval-ground-truth`: uv run python -m evaluation.retrieval_eval

Metrics, all at k = NUM_RESULTS:
hit_rate      the exact source document appears in the results
mrr           1/rank of the exact source document, averaged over queries
hit_rate_q    any answer to the same original question appears

The third one exists because the knowledge base keeps up to two answers per question.
If a query generated from the accepted answer retrieves the runner-up answer to the
same question instead, the user's need is met, but a strict document-id match scores it zero.
Reporting both keeps the strict number honest while showing the practical one.

The two strategies that call the LLM are measured on a subsample, because they issue
one request per query and a free API tier will not sit through thousands. The sample
size used is printed and stored alongside the metrics.
"""

import argparse
import random
import time  # <--- Added for rate-limit pacing and backoff
from tqdm import tqdm

from app import db, settings
from app.llm import LLMClient
from app.retrieval import STRATEGIES, RetrieverFactory
from evaluation.common import (
    GROUND_TRUTH_CSV,
    RANDOM_SEED,
    RETRIEVAL_CSV,
    markdown_table,
    read_csv,
    write_csv,
)

LLM_STRATEGIES = {"hybrid_rerank", "hybrid_rewrite"}


def evaluate(retriever, queries: list[dict], doc_to_question: dict[str, int], k: int) -> dict:
    hits = 0
    hits_by_question = 0
    reciprocal_ranks = 0.0

    for row in tqdm(queries, desc=retriever.name.ljust(15), leave=False):
        expected_doc = row["doc_id"]
        expected_question = doc_to_question.get(expected_doc)

        # Implementation of exponential backoff retry logic for LLM rate limits
        max_retries = 5
        backoff_delay = 2.0  # Initial delay in seconds
        results = None

        for attempt in range(max_retries):
            try:
                results = retriever.retrieve(row["question"], num_results=k).documents
                break  # Success, exit retry loop
            except Exception as e:
                # Check for rate limit keywords in the error message
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "429" in error_msg or "too many requests" in error_msg:
                    if attempt == max_retries - 1:
                        print(f"\n[Error] Retries exhausted for query: {row['question']}. Error: {e}")
                        raise e
                    
                    time.sleep(backoff_delay)
                    backoff_delay *= 2  # Exponential growth
                else:
                    raise e  # Re-raise any unrelated exceptions immediately

        # Preventative pacing: sleep briefly between successful LLM calls to reduce rate limit triggers
        if retriever.name in LLM_STRATEGIES:
            time.sleep(0.5)

        # Fallback safeguard if results extraction failed entirely
        if results is None:
            continue

        result_ids = [document["doc_id"] for document in results]

        if expected_doc in result_ids:
            hits += 1
            reciprocal_ranks += 1.0 / (result_ids.index(expected_doc) + 1)

        if expected_question is not None and any(
            document["question_id"] == expected_question for document in results
        ):
            hits_by_question += 1

    total = len(queries)
    return {
        "strategy": retriever.name,
        "queries": total,
        "hit_rate": hits / total,
        "mrr": reciprocal_ranks / total,
        "hit_rate_q": hits_by_question / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", type=int, default=settings.NUM_RESULTS, help="how many results to retrieve")
    parser.add_argument("--llm-sample", type=int, default=100, help="queries used for the LLM-backed strategies")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES, help="subset of strategies to evaluate")
    args = parser.parse_args()

    ground_truth = read_csv(GROUND_TRUTH_CSV)
    documents = db.load_documents()

    if not documents:
        raise RuntimeError("the documents table is empty - run `make ingest` first")

    doc_to_question = {
        document["doc_id"]: document["question_id"] for document in documents
    }

    # Drop any query whose source document is no longer in the knowledge base,
    # which happens if ingestion was re-run with different settings.
    usable = [row for row in ground_truth if row["doc_id"] in doc_to_question]
    dropped = len(ground_truth) - len(usable)
    if dropped:
        print(f"ignoring {dropped} queries whose source document is gone")

    if not usable:
        raise RuntimeError(
            f"none of the {len(ground_truth)} ground-truth queries point at a "
            f"document that is still in the knowledge base. Re-run "
            f"`make eval-ground-truth` against the current corpus."
        )

    rng = random.Random(RANDOM_SEED)
    llm_subset = usable[:]
    rng.shuffle(llm_subset)
    llm_subset = llm_subset[: args.llm_sample]

    print(
        f"evaluating {len(args.strategies)} strategies at k={args.k} "
        f"on {len(usable)} queries "
        f"({len(llm_subset)} for the LLM-backed ones)"
    )

    factory = RetrieverFactory(documents, llm=LLMClient())
    results = []

    for strategy in args.strategies:
        retriever = factory.build(strategy)
        queries = llm_subset if strategy in LLM_STRATEGIES else usable

        metrics = evaluate(retriever, queries, doc_to_question, args.k)
        metrics["k"] = args.k
        results.append(metrics)

        print(
            f"  {strategy:<16} hit_rate {metrics['hit_rate']:.3f} "
            f"mrr {metrics['mrr']:.3f} hit_rate_q {metrics['hit_rate_q']:.3f} "
            f"(n={metrics['queries']})"
        )

    columns = ["strategy", "k", "queries", "hit_rate", "mrr", "hit_rate_q"]
    write_csv(RETRIEVAL_CSV, results, columns)

    print()
    print(markdown_table(
        sorted(results, key=lambda row: row["mrr"], reverse=True),
        columns,
        {"hit_rate": ".3f", "mrr": ".3f", "hit_rate_q": ".3f"},
    ))

    best = max(results, key=lambda row: row["mrr"])
    print()
    print(f"best by MRR: {best['strategy']}")
    print(f"set RETRIEVAL_STRATEGY={best['strategy']} in .env to serve it")

    if best["strategy"] in LLM_STRATEGIES:
        print(
            "note: this one was measured on the subsample, so compare it "
            "against the others with that in mind"
        )


if __name__ == "__main__":
    main()
