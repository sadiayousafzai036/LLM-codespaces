"""Retrieval strategies, built as small composable pieces.

Seven strategies get compared in evaluation/retrieval_eval.py:

  text            minsearch keyword search, uniform field weights
  text_boosted    same, with field weights tuned for question-shaped queries
  vector          embeddings of the question side of each document
  vector_full     embeddings of question plus answer text
  hybrid          reciprocal rank fusion of text_boosted and the better vector
  hybrid_rerank   hybrid candidates re-ordered by the LLM
  hybrid_rewrite  the user's question rewritten before hybrid search

Re-ranking and rewriting are wrappers, not separate implementations, so any
base retriever can be wrapped by either. `RETRIEVAL_STRATEGY` in the
environment decides what the app serves.
"""

from dataclasses import dataclass

from minsearch import Index, VectorSearch
from pydantic import BaseModel, Field

from app.embedder import Embedder
from app.llm import EMPTY_USAGE, LLMClient, Usage

TEXT_FIELDS = ["title", "question_text", "answer_text", "tags"]
KEYWORD_FIELDS = ["primary_tag"]

# Users type questions, so the question title is the field most likely to match
# their wording. The answer body still matters but is long and noisy, and tags
# are single words that would otherwise dominate short queries.
TUNED_BOOSTS = {
    "title": 3.0,
    "question_text": 1.0,
    "answer_text": 0.6,
    "tags": 0.4,
}

# Standard RRF constant. Large enough that the top of each list does not
# completely dominate the fused ordering.
RRF_K = 60

# How many candidates the fusion and re-ranking stages work with.
FUSION_CANDIDATES = 20
RERANK_CANDIDATES = 20


def question_side_text(document: dict) -> str:
    return f"{document['title']}\n{document['question_text']}"


def full_text(document: dict) -> str:
    return (
        f"{document['title']}\n{document['question_text']}\n{document['answer_text']}"
    )


def _filter_for(tag: str | None) -> dict:
    return {"primary_tag": tag} if tag else {}


@dataclass
class Retrieval:
    """What a retriever returns: the documents plus what it cost to get them."""

    documents: list[dict]
    usage: Usage = EMPTY_USAGE
    rewritten_query: str | None = None


# --------------------------------------------------------------- base retrievers


class TextRetriever:
    def __init__(self, documents: list[dict], boosts: dict | None = None,
                 name: str = "text"):
        self.name = name
        self.boosts = boosts or {}
        self.index = Index(text_fields=TEXT_FIELDS, keyword_fields=KEYWORD_FIELDS)
        self.index.fit(documents)

    def retrieve(self, query: str, num_results: int = 5,
                 tag: str | None = None) -> Retrieval:
        results = self.index.search(
            query,
            filter_dict=_filter_for(tag),
            boost_dict=self.boosts,
            num_results=num_results,
        )
        return Retrieval(documents=results)


class VectorRetriever:
    def __init__(self, documents: list[dict], embedder: Embedder,
                 text_fn=question_side_text, name: str = "vector",
                 show_progress: bool = True):
        self.name = name
        self.embedder = embedder
        if show_progress:
            print(f"embedding {len(documents)} documents for '{name}'")
        matrix = embedder.encode_batch(
            [text_fn(document) for document in documents],
            show_progress=show_progress,
        )
        self.index = VectorSearch(keyword_fields=KEYWORD_FIELDS)
        self.index.fit(matrix, documents)

    def retrieve(self, query: str, num_results: int = 5,
                 tag: str | None = None) -> Retrieval:
        vector = self.embedder.encode(query)
        results = self.index.search(
            vector, filter_dict=_filter_for(tag), num_results=num_results
        )
        return Retrieval(documents=results)


class HybridRetriever:
    """Reciprocal rank fusion of a keyword and a vector retriever.

    RRF combines rankings rather than scores, which matters here because
    minsearch's TF-IDF scores and cosine similarities are on unrelated scales -
    normalising them against each other would be arbitrary. Each document gets
    1/(k + rank) from every list it appears in, and the sums decide the order.
    """

    def __init__(self, text: TextRetriever, vector: VectorRetriever,
                 k: int = RRF_K, candidates: int = FUSION_CANDIDATES,
                 name: str = "hybrid"):
        self.name = name
        self.text = text
        self.vector = vector
        self.k = k
        self.candidates = candidates

    def retrieve(self, query: str, num_results: int = 5,
                 tag: str | None = None) -> Retrieval:
        rankings = [
            self.text.retrieve(query, self.candidates, tag).documents,
            self.vector.retrieve(query, self.candidates, tag).documents,
        ]

        scores: dict[str, float] = {}
        by_id: dict[str, dict] = {}
        for documents in rankings:
            for rank, document in enumerate(documents, start=1):
                doc_id = document["doc_id"]
                by_id[doc_id] = document
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)

        best = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
        return Retrieval(documents=[by_id[doc_id] for doc_id in best[:num_results]])


# ------------------------------------------------------------------- wrappers


class _Ranking(BaseModel):
    doc_numbers: list[int] = Field(
        description="Candidate numbers, most relevant first"
    )


RERANK_INSTRUCTIONS = """
You rank cooking Q&A snippets by how well they answer a user's question.

Judge whether the snippet actually answers what was asked, not whether it
shares words with it. A snippet about storing bread does not answer a question
about proofing bread, however similar the wording.

Return the candidate numbers ordered from most to least relevant. Include only
candidates that are genuinely relevant; it is correct to return fewer than you
were given, and correct to return an empty list if none of them fit.
""".strip()

RERANK_PROMPT = """
User question: {query}

Candidates:
{candidates}
""".strip()


class RerankingRetriever:
    """Fetch a wide candidate set, then let the LLM order the top few.

    Keyword and vector search both rank on surface similarity. Asking a model to
    read the candidates catches the cases where the wording matches but the
    subject does not.
    """

    def __init__(self, base, llm: LLMClient, candidates: int = RERANK_CANDIDATES,
                 model: str | None = None, name: str | None = None):
        self.base = base
        self.llm = llm
        self.candidates = candidates
        self.model = model
        self.name = name or f"{base.name}_rerank"

    def retrieve(self, query: str, num_results: int = 5,
                 tag: str | None = None) -> Retrieval:
        pool = self.base.retrieve(query, self.candidates, tag).documents
        if len(pool) <= num_results:
            return Retrieval(documents=pool)

        listing = "\n\n".join(
            f"[{number}] {document['title']}\n"
            f"{document['answer_text'][:400]}"
            for number, document in enumerate(pool, start=1)
        )

        try:
            ranking, usage = self.llm.structured(
                RERANK_INSTRUCTIONS,
                RERANK_PROMPT.format(query=query, candidates=listing),
                _Ranking,
                model=self.model,
            )
        except RuntimeError as exc:
            # A re-ranker that fails should degrade to the base ordering rather
            # than take the whole request down with it.
            print(f"  rerank failed ({exc}), falling back to fused order")
            return Retrieval(documents=pool[:num_results])

        ordered = []
        seen = set()
        for number in ranking.doc_numbers:
            index = number - 1
            if 0 <= index < len(pool) and index not in seen:
                seen.add(index)
                ordered.append(pool[index])

        # If the model discarded almost everything, top up from the fused order
        # so the LLM still has context to work with.
        chosen = {document["doc_id"] for document in ordered}
        for document in pool:
            if len(ordered) >= num_results:
                break
            if document["doc_id"] not in chosen:
                ordered.append(document)
                chosen.add(document["doc_id"])

        return Retrieval(documents=ordered[:num_results], usage=usage)


class _Rewrite(BaseModel):
    query: str = Field(description="The rewritten search query")


REWRITE_INSTRUCTIONS = """
You rewrite a home cook's question into a search query for a cooking Q&A site.

Keep the meaning exactly. Replace casual phrasing with the terms an experienced
cook would use, expand what a pronoun refers to, and drop pleasantries and
filler. Add a synonym only when the original word is ambiguous.

Keep it under 20 words. If the question is already a good query, return it
unchanged.
""".strip()


class QueryRewritingRetriever:
    """Rewrite the question before searching.

    Real questions arrive as "why did my bread come out like a brick" while the
    corpus says "dense crumb" and "under-proofed". Keyword search cannot bridge
    that; a one-sentence rewrite can.
    """

    def __init__(self, base, llm: LLMClient, model: str | None = None,
                 name: str | None = None):
        self.base = base
        self.llm = llm
        self.model = model
        self.name = name or f"{base.name}_rewrite"

    def retrieve(self, query: str, num_results: int = 5,
                 tag: str | None = None) -> Retrieval:
        try:
            rewrite, usage = self.llm.structured(
                REWRITE_INSTRUCTIONS, query, _Rewrite, model=self.model
            )
            rewritten = rewrite.query.strip() or query
        except RuntimeError as exc:
            print(f"  rewrite failed ({exc}), searching with the original query")
            rewritten = query
            usage = EMPTY_USAGE

        result = self.base.retrieve(rewritten, num_results, tag)
        return Retrieval(
            documents=result.documents,
            usage=usage + result.usage,
            rewritten_query=rewritten if rewritten != query else None,
        )


# -------------------------------------------------------------------- factory

STRATEGIES = [
    "text",
    "text_boosted",
    "vector",
    "vector_full",
    "hybrid",
    "hybrid_rerank",
    "hybrid_rewrite",
]


class RetrieverFactory:
    """Builds retrievers on demand and shares the expensive parts.

    Embedding the corpus takes tens of seconds, so both vector variants and
    anything built on them reuse a single cached instance.
    """

    def __init__(self, documents: list[dict], llm: LLMClient | None = None,
                 embedder: Embedder | None = None, show_progress: bool = True):
        self.documents = documents
        self.llm = llm
        self._embedder = embedder
        self.show_progress = show_progress
        self._cache: dict[str, object] = {}

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def _require_llm(self) -> LLMClient:
        if self.llm is None:
            raise ValueError("this strategy needs an LLM client")
        return self.llm

    def build(self, strategy: str):
        if strategy in self._cache:
            return self._cache[strategy]
        if strategy not in STRATEGIES:
            raise ValueError(
                f"unknown strategy '{strategy}', expected one of {STRATEGIES}"
            )

        if strategy == "text":
            retriever = TextRetriever(self.documents, name="text")
        elif strategy == "text_boosted":
            retriever = TextRetriever(
                self.documents, boosts=TUNED_BOOSTS, name="text_boosted"
            )
        elif strategy == "vector":
            retriever = VectorRetriever(
                self.documents, self.embedder, question_side_text,
                name="vector", show_progress=self.show_progress,
            )
        elif strategy == "vector_full":
            retriever = VectorRetriever(
                self.documents, self.embedder, full_text,
                name="vector_full", show_progress=self.show_progress,
            )
        elif strategy == "hybrid":
            retriever = HybridRetriever(
                self.build("text_boosted"), self.build("vector")
            )
        elif strategy == "hybrid_rerank":
            # Check for the LLM before building the base, which embeds the whole
            # corpus. Otherwise a missing API key costs a minute of embedding
            # before it reports itself.
            llm = self._require_llm()
            retriever = RerankingRetriever(self.build("hybrid"), llm)
        else:  # hybrid_rewrite
            llm = self._require_llm()
            retriever = QueryRewritingRetriever(self.build("hybrid"), llm)

        self._cache[strategy] = retriever
        return retriever
