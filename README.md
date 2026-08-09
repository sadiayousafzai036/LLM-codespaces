# Cooking Q&A Assistant

A question-answering assistant for home cooks, built on the highest-voted
questions and answers from [Seasoned Advice](https://cooking.stackexchange.com),
the cooking Stack Exchange site. Ask it a cooking question in plain language and
it retrieves the relevant community answers, writes a single answer grounded in
them, and shows you which sources it used.

---

## Contents

- [The problem](#the-problem)
- [The data](#the-data)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Running it locally](#running-it-locally)
- [Evaluation](#evaluation)
- [Monitoring](#monitoring)
- [Best practices implemented](#best-practices-implemented)
- [Project structure](#project-structure)
- [Configuration reference](#configuration-reference)
- [Limitations and what I would do next](#limitations-and-what-i-would-do-next)
- [Licence and attribution](#licence-and-attribution)
- [Evaluation criteria map](#evaluation-criteria-map)

---

## The problem

Something goes wrong mid-cook and you need an answer in the next two minutes.
The bread came out dense. The custard split. There is cooked rice that has been
on the counter since last night and a decision to make about it.

Seasoned Advice has excellent answers to almost all of these, written by people
who know why things happen rather than just what to do. The problem is getting
at them under time pressure:

- **You have to know the vocabulary to find the answer.** The site says
  "under-proofed" and "gluten development"; you type "why is my bread like a
  brick". Keyword search does not bridge that gap, so the answer that would help
  you is invisible.
- **The answer is spread across several posts.** One answer covers proofing time,
  another covers oven spring, a third mentions that your flour might be the
  problem. You end up reading four threads and assembling the answer yourself.
- **Skimming costs time you do not have.** Accepted answers are often long, with
  the reasoning before the conclusion.
- **General-purpose chatbots will confidently invent an answer.** For "how long
  can I leave this out", a plausible-sounding guess is actively dangerous. You
  want an answer with a source you can check.

This assistant addresses those four things directly. It rewrites or re-ranks
your question against the corpus so casual phrasing still finds the right posts,
synthesises across several answers at once, returns a short answer, and cites
the posts it used so you can verify it. When the corpus does not cover your
question it says so instead of guessing — the prompt is built around that, and
the evaluation measures whether it holds.

**Who it is for:** home cooks who want a sourced answer fast. **What it is not:**
a recipe generator, and not a food-safety authority — it reports what the
community said and links you to it.

---

## The data

[Seasoned Advice](https://cooking.stackexchange.com) via the
[Stack Exchange API v2.3](https://api.stackexchange.com/docs).

| | |
|---|---|
| Source | `api.stackexchange.com/2.3`, site `cooking` |
| Selection | Top ~2,000 questions by vote score, with their answers |
| Documents | One question paired with one of its answers |
| Typical size | ~3,000–4,000 documents (varies with the answer filter) |
| Auth | None required. Anonymous quota is 300 requests/day; a free key raises it to 10,000 |
| Licence | CC BY-SA — see [Licence and attribution](#licence-and-attribution) |

**Why sort by votes rather than recency.** A fixed page budget buys far better
content this way, and the top of the list barely moves between runs, so the
knowledge base is stable enough to evaluate against.

**Document granularity.** One document is one question plus one answer. Splitting
by paragraph produced chunks that read as instructions with no subject ("let it
rest for an hour" — let *what* rest?). Keeping a whole question with all of its
answers made single documents too long and diluted retrieval. Pairing each answer
with its question title keeps every document self-contained. We keep at most two
answers per question, the accepted one plus the next highest voted; past that
they mostly repeat each other and crowd out other questions from the context.

Answers shorter than 80 characters are dropped — they are almost always "me too"
or a bare link.

**This is not the course FAQ dataset.** `project.md` rules that out, and nothing
here touches it.

---

## How it works

```
   Stack Exchange API  (questions + answers, HTML bodies)
            │
            ▼
   ┌──────────────────┐
   │  dlt pipeline    │  ingestion/pipeline.py
   │  schema inferred │  → Postgres  raw.questions, raw.answers
   └──────────────────┘
            │  HTML → text, pair up, filter, truncate
            ▼           ingestion/documents.py
   ┌──────────────────┐
   │ public.documents │  the knowledge base
   └──────────────────┘
            │
            ├──────────────► minsearch Index        (TF-IDF keyword search)
            └──────────────► minsearch VectorSearch (MiniLM ONNX embeddings)
                       │
                       ▼
              reciprocal rank fusion  ──► LLM re-rank  ──► top 5 documents
                                                              │
                                                              ▼
                                              prompt + Groq chat completion
                                                              │
                          ┌───────────────────────────────────┴────────────┐
                          ▼                                                ▼
                 Streamlit chat (:8501)                    public.conversations
                 answer + sources + 👍/👎                   public.feedback
                                                                    │
                                                                    ▼
                                                     Streamlit dashboard (:8502)
```

**Two-layer storage.** The dlt pipeline writes untouched API payloads into a
`raw` schema; a separate transform step derives `public.documents` from it. This
matters more than it looks: cleaning rules change far more often than the source
data, and re-deriving the knowledge base costs nothing, whereas re-extracting
costs API quota you may not have until tomorrow. `python -m ingestion.run
--skip-extract` rebuilds from `raw` without touching the network.

**Why ONNX instead of sentence-transformers.** Running MiniLM through
`onnxruntime` avoids pulling in PyTorch, which takes the Docker image from
roughly 2.5 GB to under 500 MB. The cost is doing mean pooling and normalisation
by hand, which is about a dozen lines in `app/embedder.py`.

**Why an LLM adapter.** The course code uses the OpenAI Responses API
(`client.responses.create`). Groq's endpoint is OpenAI-compatible but implements
only Chat Completions, so none of that transfers. `app/llm.py` is the adapter:
everything above it calls `complete()` or `structured()`, and switching provider
is two environment variables. Structured output asks for a JSON object and
validates it against a Pydantic model with a repair retry, rather than relying on
native schema enforcement, because `json_schema` support varies by model on Groq
while `json_object` plus validation works everywhere.

---

## Quick start

Everything runs in Docker. You need Docker with Compose v2 and a free
[Groq API key](https://console.groq.com/keys).

```bash
git clone <your-repo-url>
cd cooking-qa-assistant

cp .env.example .env
# open .env and set GROQ_API_KEY

docker compose up --build
```

The first build takes a few minutes: it installs dependencies and bakes the
90 MB embedding model into the image. On startup, Compose waits for Postgres to
pass its healthcheck, runs the ingestion job to completion, and only then starts
the two apps — so neither ever comes up against an empty knowledge base.

Ingestion pulls ~2,000 questions and their answers, which is about 40 API calls
and takes two or three minutes.

Then open:

| | |
|---|---|
| Chat app | <http://localhost:8501> |
| Dashboard | <http://localhost:8502> |

To stop, `docker compose down`. To also wipe the database, `docker compose down -v`.

### Sanity check

```bash
docker compose exec chat python -m app.assistant "why is my sourdough so dense?"
```

Prints the answer, the sources, timings and token counts, and logs the
conversation so it shows up on the dashboard.

---

## Running it locally

Useful for development and required for the evaluation scripts, which are
deliberately not containerised — they are long-running interactive jobs, not
services.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync                                  # install dependencies, pinned by uv.lock
uv run python scripts/download_model.py  # fetch the ONNX embedding model
docker compose up -d postgres            # database only
uv run python -m ingestion.run           # extract + build the knowledge base

uv run streamlit run app/chat.py --server.port 8501
uv run streamlit run app/dashboard.py --server.port 8502
```

`make help` lists all of these as short targets.

`POSTGRES_HOST` defaults to `localhost`, which is what you want here; Compose
overrides it to `postgres` for the containers.

### Pointing it at a different Stack Exchange site

Nothing in the code is cooking-specific. Change one variable and re-ingest:

```bash
SE_SITE=bicycles uv run python -m ingestion.run
```

`gardening`, `diy`, `coffee`, `homebrew` and `outdoors` all work the same way.
The prompts mention cooking by name, so adjust `app/rag.py` if you switch for
real.

---

## Evaluation

Two things get evaluated separately: whether retrieval finds the right document,
and whether the LLM writes a good answer from it.

### Ground truth

Hit rate and MRR need to know which document *should* come back for a query.
Hand-labelling hundreds of pairs is not realistic, so we generate the question
set instead: for a stratified sample of documents, the LLM writes questions a
home cook might type that this particular answer would resolve. The document
that produced a question is that question's relevant result.

```bash
uv run python -m evaluation.ground_truth --documents 120 --per-document 3
```

Around 360 questions across as many distinct tags as the corpus allows. The
sample is round-robin across tags rather than flat random, because the corpus is
heavily skewed toward a few popular tags and a flat sample would mostly measure
performance on those.

**The shortcut this deliberately avoids.** It is tempting to use each question's
own Stack Exchange title as the query. That inflates the results badly — titles
are indexed verbatim, so keyword search scores near-perfectly and the comparison
between strategies stops meaning anything. Generated paraphrases keep it honest,
and the prompt explicitly forbids reusing the title's wording.

### Retrieval

```bash
uv run python -m evaluation.retrieval_eval --strategies text text_boosted vector vector_full hybrid
```

Seven strategies, all at k=5:

| Strategy | What it does |
|---|---|
| `text` | minsearch TF-IDF, uniform field weights — the baseline |
| `text_boosted` | same, weights tuned toward the question title |
| `vector` | MiniLM embeddings of title + question body |
| `vector_full` | MiniLM embeddings of title + question + answer body |
| `hybrid` | reciprocal rank fusion of `text_boosted` and `vector` |
| `hybrid_rerank` | hybrid's top 20, re-ordered by the LLM |
| `hybrid_rewrite` | question rewritten by the LLM, then hybrid |

Three metrics: `hit_rate` (the exact source document is in the top 5), `mrr`
(1/rank of it, averaged), and `hit_rate_q` (any answer to the same original
question is in the top 5). The third exists because the knowledge base keeps up
to two answers per question — if a query written from the accepted answer
retrieves the runner-up to the same question, the user's need is met but a strict
id match scores it zero. Reporting both keeps the strict number honest and shows
the practical one.

**Why RRF rather than blending scores.** minsearch's TF-IDF scores and cosine
similarities live on unrelated scales, so any weighting between them would be
arbitrary. RRF combines *rankings*: each document collects `1/(60 + rank)` from
every list it appears in.

The two LLM-backed strategies are measured on a 100-query subsample, since they
issue one request per query and a free tier will not sit through thousands. The
sample size is printed and stored in the CSV alongside the metrics.

#### Results
Raw output: `evaluation/results/retrieval_metrics.csv`.

### LLM answer quality

```bash
uv run python -m evaluation.rag_eval --questions 20
```

Three prompt variants against two models, judged by an LLM:

| Variant | What it does |
|---|---|
| `concise` | minimal instructions — the naive baseline |
| `grounded` | refuses to go beyond the context, cites sources, admits gaps |
| `structured` | grounded, plus a fixed short answer / details / caveats shape |

Models: `LLM_MODEL` and `LLM_MODEL_FAST` from `.env`.

The judge scores two things per answer. **Relevance** — does it address the
question. **Groundedness** — is every claim supported by the retrieved context.
Relevance alone is not enough for a RAG system: an answer can be perfectly
on-topic and entirely invented, which is exactly the failure that matters when
the subject is food safety. Both come back from a single call, which also halves
the request count. The reported `score` weights them equally, so a variant cannot
win on relevance while making things up.

Retrieval is held fixed at `hybrid` here, even though the app serves
`hybrid_rerank`. The point is to isolate the effect of the prompt and the model,
and re-ranking would add a third LLM call per answer to a script that already
makes two.

Judge failures are recorded as `JUDGE_FAILED` and excluded from the rates rather
than being silently counted as passes.

**Cost.** The default is 20 questions × 3 variants × 2 models = 120 answers plus
120 judgements. On a free tier expect this to take a while and to hit rate limits
that the client backs off from. Lower `--questions` for a smoke test.

#### Results
 `evaluation/results/rag_answers.csv`.


---

## Monitoring

Every answered question writes a row to `public.conversations` — question,
answer, strategy, prompt variant, model, retrieved document ids, token counts,
retrieval and total latency, estimated cost. Feedback goes to `public.feedback`,
from two sources: the user's 👍/👎 in the chat app, and the LLM judge when it is
enabled in the sidebar.

The dashboard at <http://localhost:8502> has eight charts:

1. **Questions per hour** — usage over time
2. **Response time over time** — total vs retrieval-only, so a slowdown can be
   attributed
3. **Token usage per hour** — prompt vs completion, stacked
4. **User feedback** — 👍 against 👎
5. **Judge: relevance** — RELEVANT / PARTLY_RELEVANT / NON_RELEVANT
6. **Judge: groundedness** — SUPPORTED / PARTLY_SUPPORTED / UNSUPPORTED
7. **Most asked-about topics** — by the dominant tag of the retrieved documents
8. **Response time distribution** — histogram, which surfaces tail latency that
   the average hides

Plus a KPI row (conversations, average response and retrieval time, tokens,
estimated cost, thumbs-up rate), a per-strategy table comparing latency, tokens
and thumbs by strategy/prompt/model, and the 25 most recent questions with their
verdicts.

Charts bucket by hour rather than day: a fresh database holds a few dozen rows
from one session, and daily buckets would collapse all of it into a single bar.

Cost is an estimate. Groq's free tier costs nothing, so
`PRICE_INPUT_PER_MTOK` and `PRICE_OUTPUT_PER_MTOK` default to 0 and the chart
reads zero — token counts are the real signal. Set those two variables to the
published rates if you move to a paid plan.

---

## Best practices implemented

**Hybrid search** — `app/retrieval.py:HybridRetriever`. Reciprocal rank fusion
of TF-IDF keyword search and MiniLM vector search. Evaluated against both of its
own components plus a second embedding variant.

**Document re-ranking** — `app/retrieval.py:RerankingRetriever`. Takes hybrid's
top 20 and has the LLM order them by whether they actually answer the question
rather than whether they share words with it. It is explicitly allowed to return
fewer candidates than it was given. If the re-rank call fails, it degrades to the
fused order instead of taking the request down.

**User query rewriting** — `app/retrieval.py:QueryRewritingRetriever`. Rewrites
"why did my bread come out like a brick" into corpus vocabulary before searching.
The rewritten query is shown in the UI and stored in `conversations.rewritten_query`,
so you can see what it actually searched for. Falls back to the original query on
failure.

All three are wrappers rather than separate implementations, so any base
retriever can be composed with either, and all of them appear in the retrieval
comparison.

---

## Project structure

```
cooking-qa-assistant/
├── ingestion/
│   ├── stackexchange.py   API client: pagination, backoff, quota
│   ├── pipeline.py        dlt pipeline → Postgres raw schema
│   ├── html_text.py       post HTML → plain text (stdlib only)
│   ├── documents.py       raw tables → public.documents
│   └── run.py             entry point, --skip-extract to rebuild offline
├── app/
│   ├── settings.py        all configuration, read from the environment
│   ├── llm.py             Groq/OpenAI adapter, structured output with retry
│   ├── embedder.py        MiniLM via onnxruntime
│   ├── retrieval.py       seven strategies, composed from small pieces
│   ├── rag.py             prompt variants, context building, the flow
│   ├── judge.py           LLM-as-judge: relevance + groundedness
│   ├── db.py              schema, writes, reads
│   ├── assistant.py       wiring, plus a CLI entry point
│   ├── chat.py            Streamlit chat UI
│   └── dashboard.py       Streamlit monitoring dashboard
├── evaluation/
│   ├── ground_truth.py    generate the question set
│   ├── retrieval_eval.py  hit rate and MRR across strategies
│   ├── rag_eval.py        judge prompt variants × models
│   ├── common.py          shared helpers
│   └── results/           committed CSV output
├── tests/                 unit tests, no network or database needed
├── scripts/download_model.py
├── docker-compose.yml     postgres + ingest + chat + dashboard
├── Dockerfile
├── Makefile
└── pyproject.toml         + uv.lock for exact versions
```

### Tests

```bash
make test        # or: uv run pytest
```

Around 40 tests over the pieces where a silent regression would be expensive:
the HTML cleaner, the rank-fusion arithmetic and its failure modes, context
assembly, and token accounting. They use stub retrievers and stub LLM clients, so
they need no API key, no database and no embedding model, and run in under a
second.

---

## Configuration reference

Everything is an environment variable with a working default; only
`GROQ_API_KEY` is required. See `.env.example` for the annotated list.

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | — | Required. <https://console.groq.com/keys> |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Any OpenAI-compatible endpoint |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Check <https://console.groq.com/docs/models> — Groq retires model ids |
| `LLM_MODEL_FAST` | `llama-3.1-8b-instant` | The comparison model in the RAG eval |
| `JUDGE_MODEL` | same as `LLM_MODEL` | |
| `RETRIEVAL_STRATEGY` | `hybrid_rerank` | Set from the retrieval evaluation |
| `PROMPT_VARIANT` | `grounded` | Set from the RAG evaluation |
| `NUM_RESULTS` | `5` | Documents passed to the LLM |
| `SE_SITE` | `cooking` | Any Stack Exchange site slug |
| `SE_MAX_PAGES` | `20` | 100 questions per page |
| `SE_API_KEY` | — | Optional, raises the daily quota to 10,000 |
| `PRICE_*_PER_MTOK` | `0.0` | Only used for the cost estimate |
| `POSTGRES_*` | see `.env.example` | Compose overrides the host |

---

## Limitations and what I would do next

- **Model ids drift.** Groq retires model names on its own schedule. If a call
  400s on an unknown model, check the current list and update `LLM_MODEL`.
- **The ground truth is LLM-generated.** It is consistent and cheap, but it
  inherits the generating model's idea of how people phrase questions. Real query
  logs would be better; there are none yet, which is partly what the monitoring
  is for.
- **The judge and the answerer are the same model family.** Self-preference bias
  is a known effect, so the absolute rates are probably a little optimistic. The
  *comparison between variants* is the part worth trusting.
- **Retrieval is in-memory.** minsearch rebuilds the index and re-embeds the
  corpus at startup, which is fine at a few thousand documents and would not be
  at a few hundred thousand. pgvector is the obvious next step, and Postgres is
  already there.
- **No conversation memory.** Each question is answered independently; follow-ups
  like "what about at room temperature?" do not work.
- **Only two answers per question are indexed**, so a minority view buried in a
  fourth answer is invisible to the assistant.
- **Not deployed.** It runs locally under Compose. The image is self-contained,
  so a container host would be straightforward.

---

## Licence and attribution

Content from Stack Exchange is licensed
[CC BY-SA](https://stackoverflow.com/help/licensing), which requires attribution.
This project keeps that obligation rather than stripping it:

- Every document stores the answer author, the question author and a link to the
  original post.
- The chat UI shows the attribution line and a link for every source it used.
- The prompt instructs the model to cite sources by number, and those numbers map
  to the links shown in the UI.

No Stack Exchange content is redistributed in this repository — the pipeline
fetches it from the public API at run time.

The code in this repository is available under the MIT licence.

---

## Evaluation criteria map

Against the criteria in the course's `project.md`, for reviewers:

| Criterion | Where | Notes |
|---|---|---|
| Problem description | [The problem](#the-problem) | Who it is for, four specific failures it addresses |
| Retrieval flow | `app/rag.py`, `app/retrieval.py` | Postgres-backed knowledge base + LLM, both in the flow |
| Retrieval evaluation | [Retrieval](#retrieval) | Seven strategies compared on hit rate and MRR; the winner is what `RETRIEVAL_STRATEGY` serves |
| LLM evaluation | [LLM answer quality](#llm-answer-quality) | Three prompts × two models, judged on relevance *and* groundedness; the winner is what `PROMPT_VARIANT` serves |
| Interface | `app/chat.py` | Streamlit UI, plus a CLI in `app/assistant.py` |
| Ingestion pipeline | `ingestion/pipeline.py` | Automated with dlt, into a Postgres raw schema |
| Monitoring | `app/dashboard.py` | User feedback collected, plus a dashboard with 8 charts |
| Containerisation | `docker-compose.yml` | Database, ingestion job and both apps, with ordered startup |
| Reproducibility | [Quick start](#quick-start) | One command; dataset fetched from a public API needing no auth; versions pinned in `uv.lock` |
| Hybrid search | `app/retrieval.py:HybridRetriever` | RRF over keyword + vector, evaluated against both components |
| Document re-ranking | `app/retrieval.py:RerankingRetriever` | LLM re-ranks the top 20 |
| Query rewriting | `app/retrieval.py:QueryRewritingRetriever` | Rewritten query stored and displayed |
