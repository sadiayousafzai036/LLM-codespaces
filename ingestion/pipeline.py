"""dlt pipeline: Stack Exchange API -> Postgres `raw` schema.

Why dlt rather than a hand-rolled loader: it infers and evolves the table
schema from the payloads, handles typing and normalisation of column names,
and gives us a load trace we can inspect when something looks off.

Extraction is deliberately eager. We need the full list of question ids before
we can ask for their answers (the answers endpoint takes up to 100 ids per
call), and a couple of thousand dictionaries costs nothing to hold in memory.
Doing it this way keeps the ordering explicit instead of relying on the order
dlt happens to evaluate resources in.
"""

import dlt

from app import settings
from ingestion.stackexchange import StackExchangeClient

RAW_DATASET = "raw"
PIPELINE_NAME = "stackexchange"


def build_pipeline() -> dlt.Pipeline:
    return dlt.pipeline(
        pipeline_name=PIPELINE_NAME,
        destination=dlt.destinations.postgres(credentials=settings.postgres_dsn()),
        dataset_name=RAW_DATASET,
    )


def load_raw(
    site: str | None = None,
    max_pages: int | None = None,
    api_key: str | None = None,
) -> dict:
    site = site or settings.SE_SITE
    max_pages = max_pages if max_pages is not None else settings.SE_MAX_PAGES
    api_key = api_key if api_key is not None else settings.SE_API_KEY

    client = StackExchangeClient(site=site, api_key=api_key)

    print(f"fetching up to {max_pages} pages of questions from {site}.stackexchange.com")
    questions = list(client.iter_questions(max_pages))
    print(f"got {len(questions)} questions")

    question_ids = [question["question_id"] for question in questions]
    print(f"fetching answers for {len(question_ids)} questions")
    answers = list(client.iter_answers(question_ids))
    print(f"got {len(answers)} answers")

    if not questions:
        raise RuntimeError(
            "the API returned no questions - check SE_SITE and the daily quota"
        )

    pipeline = build_pipeline()

    # `replace` because the knowledge base is a snapshot of the top-voted posts.
    # Switch to write_disposition="merge" with these primary keys if you later
    # want to accumulate across runs.
    info = pipeline.run(
        [
            dlt.resource(
                questions,
                name="questions",
                primary_key="question_id",
                write_disposition="replace",
            ),
            dlt.resource(
                answers,
                name="answers",
                primary_key="answer_id",
                write_disposition="replace",
            ),
        ]
    )

    print(info)
    return {"questions": len(questions), "answers": len(answers)}


if __name__ == "__main__":
    load_raw()
