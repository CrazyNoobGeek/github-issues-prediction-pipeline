from __future__ import annotations

from pathlib import Path
import pandas as pd


def find_dataset_parquet() -> Path:
    here = Path.cwd()
    repo_root = Path(__file__).resolve().parents[2]

    candidates = [
        here / "ml" / "artifacts" / "datasets" / "issues_clean_latest.parquet",
        here / "artifacts" / "datasets" / "issues_clean_latest.parquet",
        repo_root / "ml" / "artifacts" / "datasets" / "issues_clean_latest.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    msg = "\n".join([f" - {p}" for p in candidates])
    raise FileNotFoundError(
        "Could not find issues_clean_latest.parquet. Looked in:\n" + msg
        + "\n\nRun the dataset build first: python -m ml.data.build_dataset"
    )


ds_path = find_dataset_parquet()
df = pd.read_parquet(ds_path)

print("Dataset:", ds_path)
print("Number of rows:", len(df))
print("\nColumns:")
for c in df.columns:
    print("-", c)