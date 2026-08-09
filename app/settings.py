"""Central place for configuration read from the environment.

Everything the app needs is an env var with a sane default, so the same code
runs locally (`.env` file) and inside Docker Compose (`environment:` block).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else float(raw)


# ----------------------------------------------------------------- LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_MODEL_FAST = os.getenv("LLM_MODEL_FAST", "llama-3.1-8b-instant")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)

PRICE_INPUT_PER_MTOK = _env_float("PRICE_INPUT_PER_MTOK", 0.0)
PRICE_OUTPUT_PER_MTOK = _env_float("PRICE_OUTPUT_PER_MTOK", 0.0)

# ----------------------------------------------------------- Retrieval
RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "hybrid_rerank")
NUM_RESULTS = _env_int("NUM_RESULTS", 5)
PROMPT_VARIANT = os.getenv("PROMPT_VARIANT", "grounded")

# ------------------------------------------------------------ Postgres
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = _env_int("POSTGRES_PORT", 5432)
POSTGRES_DB = os.getenv("POSTGRES_DB", "cooking_assistant")
POSTGRES_USER = os.getenv("POSTGRES_USER", "cooking")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "cooking")

# ---------------------------------------------- Stack Exchange source
SE_SITE = os.getenv("SE_SITE", "cooking")
SE_MAX_PAGES = _env_int("SE_MAX_PAGES", 20)
SE_API_KEY = os.getenv("SE_API_KEY", "")

# ------------------------------------------------------------ Embedder
EMBEDDING_MODEL_REPO = os.getenv("EMBEDDING_MODEL_REPO", "Xenova/all-MiniLM-L6-v2")
EMBEDDING_MODEL_DIR = os.getenv("EMBEDDING_MODEL_DIR", "models")


def embedding_model_path() -> Path:
    """Directory holding tokenizer.json and model.onnx for the embedder."""
    base = Path(EMBEDDING_MODEL_DIR)
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    return base / EMBEDDING_MODEL_REPO


def postgres_dsn() -> str:
    """libpq connection string, also understood by dlt's postgres destination."""
    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )


def require_api_key() -> str:
    """Fail loudly and early instead of deep inside an HTTP call."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add a key "
            "from https://console.groq.com/keys"
        )
    return GROQ_API_KEY
