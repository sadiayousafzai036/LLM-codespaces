"""Sentence embeddings via ONNX Runtime.

Adapted from the ONNX embedder in module 2 of the course. Running the model
through `onnxruntime` instead of `sentence-transformers` keeps PyTorch out of
the dependency tree, which takes the Docker image from roughly 2.5 GB down to
under 500 MB and makes a cold container start noticeably faster. The tradeoff is
that we do mean pooling and normalisation ourselves, which is a dozen lines.

Model: Xenova/all-MiniLM-L6-v2, 384 dimensions.
Download it first with `python scripts/download_model.py`.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from app import settings

# Large enough to keep the CPU busy, small enough that padding waste stays low.
BATCH_SIZE = 64


class Embedder:
    def __init__(self, model_path: Path | str | None = None):
        path = Path(model_path) if model_path else settings.embedding_model_path()
        tokenizer_file = path / "tokenizer.json"
        model_file = path / "model.onnx"

        if not model_file.exists():
            raise FileNotFoundError(
                f"no ONNX model at {model_file}. "
                f"Run: python scripts/download_model.py"
            )

        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.session = ort.InferenceSession(
            str(model_file), providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            vectors.append(self._encode_one_batch(batch))
            if show_progress:
                done = min(start + BATCH_SIZE, len(texts))
                print(f"\r  embedded {done}/{len(texts)}", end="", flush=True)
        if show_progress:
            print()

        return np.vstack(vectors)

    def _encode_one_batch(self, texts: list[str]) -> np.ndarray:
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=256)
        encoded = self.tokenizer.encode_batch(texts)

        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )

        hidden = self.session.run(None, feed)[0]

        # Mean-pool over real tokens only, so padding does not drag vectors
        # toward each other.
        mask = feed["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)

        # Normalising means a dot product is cosine similarity, which is what
        # minsearch's VectorSearch scores with.
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-12)).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self.session.get_outputs()[0].shape[-1]
