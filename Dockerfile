# One image serves all three roles - ingestion, chat and dashboard - because
# they share the same code and dependencies. Compose just overrides the command.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Dependencies first, so editing application code does not invalidate the
# install layer. The glob makes uv.lock optional: with it, the build is exactly
# reproducible; without it, uv resolves fresh instead of failing.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app

COPY scripts/ scripts/
COPY app/ app/
COPY ingestion/ ingestion/
COPY evaluation/ evaluation/

# Now that the packages exist, install the project itself. This is the second
# half of the standard two-step uv pattern and is what makes `from app import
# db` resolve under `streamlit run app/chat.py`, which otherwise only puts
# app/ on sys.path.
RUN uv sync --no-dev

# Bake the 90 MB ONNX embedding model into the image. Containers then start
# without reaching Hugging Face, which keeps startup fast and offline-safe. The
# cost is a slower build the first time.
RUN python scripts/download_model.py

EXPOSE 8501

CMD ["streamlit", "run", "app/chat.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
