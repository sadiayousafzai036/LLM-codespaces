.PHONY: help install model ingest db chat dashboard test eval-ground-truth eval-retrieval eval-rag up down logs clean

help:
	@echo "Local development:"
	@echo "  make install            install dependencies with uv"
	@echo "  make model              download the ONNX embedding model"
	@echo "  make db                 start only Postgres (in Docker)"
	@echo "  make ingest             run the dlt pipeline and build documents"
	@echo "  make chat               run the Streamlit chat app on :8501"
	@echo "  make dashboard          run the Streamlit dashboard on :8502"
	@echo "  make test               run the unit tests (no network or DB needed)"
	@echo ""
	@echo "Evaluation (needs ingest to have run):"
	@echo "  make eval-ground-truth  generate the ground truth question set"
	@echo "  make eval-retrieval     compare retrieval strategies (hit rate, MRR)"
	@echo "  make eval-rag           compare prompts and models with an LLM judge"
	@echo ""
	@echo "Everything in Docker:"
	@echo "  make up                 build and start the whole stack"
	@echo "  make down               stop the stack"
	@echo "  make logs               tail logs"

install:
	uv sync

model:
	uv run python scripts/download_model.py

db:
	docker compose up -d postgres

ingest:
	uv run python -m ingestion.run

chat:
	uv run streamlit run app/chat.py --server.port 8501

dashboard:
	uv run streamlit run app/dashboard.py --server.port 8502

test:
	uv run pytest

eval-ground-truth:
	uv run python -m evaluation.ground_truth

eval-retrieval:
	uv run python -m evaluation.retrieval_eval

eval-rag:
	uv run python -m evaluation.rag_eval

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker compose down -v
