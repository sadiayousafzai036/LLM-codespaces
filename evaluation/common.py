"""Shared helpers for the evaluation scripts."""

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
GROUND_TRUTH_CSV = RESULTS_DIR / "ground_truth.csv"
RETRIEVAL_CSV = RESULTS_DIR / "retrieval_metrics.csv"
RAG_CSV = RESULTS_DIR / "rag_metrics.csv"
RAG_ANSWERS_CSV = RESULTS_DIR / "rag_answers.csv"

# Fixed so that re-running any script picks the same sample and the numbers in
# the README stay comparable between runs.
RANDOM_SEED = 42


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty file to {path}")
    ensure_results_dir()
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path.relative_to(path.parents[2])}")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing - run `make eval-ground-truth` first"
        )
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def markdown_table(rows: list[dict], columns: list[str],
                   formats: dict[str, str] | None = None) -> str:
    """Render results as a markdown table, ready to paste into the README."""
    formats = formats or {}

    def cell(row: dict, column: str) -> str:
        value = row.get(column, "")
        spec = formats.get(column)
        if spec and isinstance(value, (int, float)):
            return format(value, spec)
        return str(value)

    widths = {
        column: max(len(column), *(len(cell(row, column)) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = " | ".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(cell(row, column).ljust(widths[column]) for column in columns)
        for row in rows
    ]
    return "\n".join([f"| {header} |", f"| {divider} |"] + [f"| {line} |" for line in body])
