"""Thin client for the Stack Exchange API (v2.3).

The API has three quirks that shaped this code:

1. Errors arrive as HTTP 400 with a JSON body, not as a useful status text,
   so we parse the body before deciding whether the call failed.
2. The envelope may contain a `backoff` field. Ignoring it gets your IP
   throttled, so we sleep for as long as we're told.
3. Anonymous callers get 300 requests per day per IP. Passing a free API key
   raises that to 10,000. We log the remaining quota so a failing ingest is
   easy to diagnose.

Docs: https://api.stackexchange.com/docs
"""

import time
from datetime import datetime, timezone
from typing import Iterator

import requests

API_BASE = "https://api.stackexchange.com/2.3"

# The API caps pagesize at 100, which is also the cap on how many question
# ids we can pass to /questions/{ids}/answers in one call.
PAGE_SIZE = 100
MAX_IDS_PER_CALL = 100

# `withbody` is a built-in filter that adds the HTML body of questions and
# answers to the default fields. Without it we would only get titles.
BODY_FILTER = "withbody"


class StackExchangeError(RuntimeError):
    """Raised when the API reports an error in the response envelope."""


def _to_utc(unix_seconds) -> datetime | None:
    if unix_seconds is None:
        return None
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc)


def _chunked(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class StackExchangeClient:
    def __init__(self, site: str, api_key: str = "", session=None):
        self.site = site
        self.api_key = api_key
        self.session = session or requests.Session()
        self.quota_remaining: int | None = None

    def _get(self, path: str, **params) -> dict:
        params = {"site": self.site, "pagesize": PAGE_SIZE, **params}
        if self.api_key:
            params["key"] = self.api_key

        response = self.session.get(f"{API_BASE}{path}", params=params, timeout=30)

        # Transient upstream problems are worth one retry before giving up.
        if response.status_code in (500, 502, 503, 504):
            time.sleep(5)
            response = self.session.get(f"{API_BASE}{path}", params=params, timeout=30)

        try:
            payload = response.json()
        except ValueError as exc:
            raise StackExchangeError(
                f"{path} returned non-JSON body (HTTP {response.status_code})"
            ) from exc

        if "error_id" in payload:
            raise StackExchangeError(
                f"{path} failed: {payload.get('error_name')} - "
                f"{payload.get('error_message')}"
            )

        self.quota_remaining = payload.get("quota_remaining")

        # The API tells us when we've been too eager. Respect it.
        backoff = payload.get("backoff")
        if backoff:
            print(f"  api asked us to back off for {backoff}s")
            time.sleep(backoff + 1)

        return payload

    def paginate(self, path: str, **params) -> Iterator[list[dict]]:
        """Yield one page of items at a time until the API says there's no more."""
        page = 1
        while True:
            payload = self._get(path, page=page, **params)
            yield payload.get("items", [])
            if not payload.get("has_more"):
                return
            page += 1

    def iter_questions(self, max_pages: int) -> Iterator[dict]:
        """Highest-voted questions first, so we index the most useful content.

        Sorting by votes rather than recency is deliberate: a fixed page budget
        buys far better answers this way, and the top of the list is stable
        between runs.
        """
        pages = self.paginate(
            "/questions", order="desc", sort="votes", filter=BODY_FILTER
        )
        for page_number, items in enumerate(pages, start=1):
            print(
                f"  questions page {page_number}/{max_pages} "
                f"({len(items)} items, quota left {self.quota_remaining})"
            )
            for item in items:
                yield self._question_record(item)
            if page_number >= max_pages:
                return

    def iter_answers(self, question_ids: list[int]) -> Iterator[dict]:
        """Answers for the given question ids, 100 questions per request."""
        batches = list(_chunked(question_ids, MAX_IDS_PER_CALL))
        for batch_number, batch in enumerate(batches, start=1):
            ids = ";".join(str(qid) for qid in batch)
            pages = self.paginate(
                f"/questions/{ids}/answers",
                order="desc",
                sort="votes",
                filter=BODY_FILTER,
            )
            count = 0
            for items in pages:
                for item in items:
                    count += 1
                    yield self._answer_record(item)
            print(
                f"  answers batch {batch_number}/{len(batches)} "
                f"({count} answers, quota left {self.quota_remaining})"
            )

    def _question_record(self, item: dict) -> dict:
        owner = item.get("owner") or {}
        tags = item.get("tags") or []
        return {
            "question_id": item["question_id"],
            "title": item.get("title", ""),
            "body_html": item.get("body", ""),
            # Flattened to a space-separated string on purpose. We only ever use
            # tags as search tokens and as a filter, so a child table would add
            # a join for no benefit.
            "tags": " ".join(tags),
            "primary_tag": tags[0] if tags else "untagged",
            "score": item.get("score", 0),
            "view_count": item.get("view_count", 0),
            "answer_count": item.get("answer_count", 0),
            "accepted_answer_id": item.get("accepted_answer_id"),
            "link": item.get("link", ""),
            # Kept for CC BY-SA attribution, see the licence note in the README.
            "owner_display_name": owner.get("display_name"),
            "content_license": item.get("content_license"),
            "creation_date": _to_utc(item.get("creation_date")),
            "site": self.site,
        }

    def _answer_record(self, item: dict) -> dict:
        owner = item.get("owner") or {}
        return {
            "answer_id": item["answer_id"],
            "question_id": item["question_id"],
            "body_html": item.get("body", ""),
            "score": item.get("score", 0),
            "is_accepted": bool(item.get("is_accepted")),
            "owner_display_name": owner.get("display_name"),
            "content_license": item.get("content_license"),
            "creation_date": _to_utc(item.get("creation_date")),
            "site": self.site,
        }
