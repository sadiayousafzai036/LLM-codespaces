"""Download the ONNX embedding model from the Hugging Face Hub.

Run once before starting the app, or let the Docker build do it. Files land in
$EMBEDDING_MODEL_DIR/<repo>/ as tokenizer.json and model.onnx.
"""

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

from app import settings  # noqa: E402

# Repos disagree on where they put the exported graph.
ONNX_CANDIDATES = ["onnx/model.onnx", "onnx/model_quantized.onnx", "model.onnx"]


def download(repo: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    available = list_repo_files(repo_id=repo)
    onnx_file = next((c for c in ONNX_CANDIDATES if c in available), None)
    if onnx_file is None:
        raise FileNotFoundError(f"{repo} has no ONNX export among {ONNX_CANDIDATES}")

    wanted = [("tokenizer.json", "tokenizer.json"), (onnx_file, "model.onnx")]

    # Models above 2 GB keep their weights in a sidecar file.
    external_weights = onnx_file + "_data"
    if external_weights in available:
        wanted.append((external_weights, "model.onnx_data"))

    for remote_name, local_name in wanted:
        target = destination / local_name
        if target.exists():
            print(f"  already have {target}")
            continue
        cached = hf_hub_download(repo_id=repo, filename=remote_name)
        shutil.copy2(cached, target)
        print(f"  saved {target}")


if __name__ == "__main__":
    repo = settings.EMBEDDING_MODEL_REPO
    destination = settings.embedding_model_path()
    print(f"downloading {repo} into {destination}")
    download(repo, destination)
    print("done")
