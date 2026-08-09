"""Tests for rank fusion and the retriever wrappers.

Stub retrievers stand in for the real ones so these run in milliseconds and
without a database, an embedding model or an API key. What is being tested is the
ranking arithmetic and the failure behaviour, neither of which needs real search.
"""

import pytest

from app.retrieval import HybridRetriever, Retrieval


def document(doc_id: str, **overrides) -> dict:
    base = {
        "doc_id": doc_id,
        "question_id": int(doc_id.lstrip("d")),
        "title": f"title {doc_id}",
        "question_text": "q",
        "answer_text": "a",
        "tags": "baking",
        "primary_tag": "baking",
        "question_score": 1,
        "answer_score": 1,
        "is_accepted": False,
        "link": f"https://example.com/{doc_id}",
        "attribution": "someone",
    }
    base.update(overrides)
    return base


class StubRetriever:
    def __init__(self, doc_ids, name="stub"):
        self.name = name
        self.documents = [document(doc_id) for doc_id in doc_ids]
        self.calls = []

    def retrieve(self, query, num_results=5, tag=None):
        self.calls.append((query, num_results, tag))
        return Retrieval(documents=self.documents[:num_results])


def ids(retrieval):
    return [d["doc_id"] for d in retrieval.documents]


def test_fusion_ranks_agreement_above_either_list_alone():
    # d2 is second in both lists; d1 and d3 are first in only one each.
    text = StubRetriever(["d1", "d2"])
    vector = StubRetriever(["d3", "d2"])
    hybrid = HybridRetriever(text, vector)

    result = hybrid.retrieve("query", num_results=3)

    assert ids(result)[0] == "d2"


def test_fusion_deduplicates_documents_seen_in_both_lists():
    text = StubRetriever(["d1", "d2"])
    vector = StubRetriever(["d2", "d1"])
    hybrid = HybridRetriever(text, vector)

    result = hybrid.retrieve("query", num_results=5)

    assert sorted(ids(result)) == ["d1", "d2"]


def test_fusion_respects_num_results():
    text = StubRetriever(["d1", "d2", "d3", "d4"])
    vector = StubRetriever(["d5", "d6", "d7", "d8"])
    hybrid = HybridRetriever(text, vector)

    assert len(hybrid.retrieve("query", num_results=3).documents) == 3


def test_fusion_widens_the_candidate_pool_before_narrowing():
    """Both sides must be asked for `candidates`, not for `num_results`.

    Fusing two top-5 lists throws away exactly the mid-ranked documents that
    agreement is supposed to promote.
    """
    text = StubRetriever(["d1", "d2", "d3"])
    vector = StubRetriever(["d3", "d2", "d1"])
    hybrid = HybridRetriever(text, vector, candidates=20)

    hybrid.retrieve("query", num_results=2)

    assert text.calls[0][1] == 20
    assert vector.calls[0][1] == 20


def test_fusion_passes_the_tag_filter_through():
    text = StubRetriever(["d1"])
    vector = StubRetriever(["d2"])
    hybrid = HybridRetriever(text, vector)

    hybrid.retrieve("query", num_results=2, tag="food-safety")

    assert text.calls[0][2] == "food-safety"
    assert vector.calls[0][2] == "food-safety"


def test_fusion_handles_one_empty_side():
    text = StubRetriever([])
    vector = StubRetriever(["d1", "d2"])
    hybrid = HybridRetriever(text, vector)

    assert ids(hybrid.retrieve("query", num_results=5)) == ["d1", "d2"]


def test_fusion_of_two_empty_sides_returns_nothing():
    hybrid = HybridRetriever(StubRetriever([]), StubRetriever([]))
    assert hybrid.retrieve("query").documents == []


@pytest.mark.parametrize("k", [1, 60, 200])
def test_larger_k_flattens_the_influence_of_rank(k):
    """Sanity check on the RRF constant: agreement should still win."""
    text = StubRetriever(["d1", "d2"])
    vector = StubRetriever(["d3", "d2"])
    hybrid = HybridRetriever(text, vector, k=k)

    assert ids(hybrid.retrieve("query", num_results=3))[0] == "d2"
