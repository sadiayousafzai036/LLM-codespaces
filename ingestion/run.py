"""Entry point for ingestion: API -> raw tables -> knowledge base.

    python -m ingestion.run                  full run
    python -m ingestion.run --skip-extract   rebuild documents only, no API calls

The second form matters in practice. Cleaning rules change more often than the
source data, and the anonymous Stack Exchange quota is 300 requests a day, so
being able to re-derive the knowledge base from the raw tables for free is the
difference between iterating freely and waiting until tomorrow.
"""

import argparse

from app import db
from ingestion import documents, pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="rebuild documents from the existing raw tables without calling the API",
    )
    parser.add_argument("--site", help="Stack Exchange site slug, e.g. cooking")
    parser.add_argument(
        "--max-pages", type=int, help="pages of 100 questions to fetch"
    )
    args = parser.parse_args()

    db.wait_until_ready()
    db.init_schema()

    if args.skip_extract:
        print("skipping extraction, reusing the raw schema")
    else:
        pipeline.load_raw(site=args.site, max_pages=args.max_pages)

    count = documents.rebuild()
    print(f"\nknowledge base ready: {count} documents")


if __name__ == "__main__":
    main()
