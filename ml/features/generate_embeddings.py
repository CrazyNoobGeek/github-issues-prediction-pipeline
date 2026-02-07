# ml/features/generate_embeddings.py

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features.embed_text import embed_texts, save_embeddings


def _safe_model_tag(model_name: str) -> str:
    # sentence-transformers/all-MiniLM-L6-v2 -> all-MiniLM-L6-v2
    tag = model_name.split("/")[-1]
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in tag)


def main():
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "ml/artifacts")).resolve()
    ds_path = artifacts_dir / "datasets" / "issues_clean_latest.parquet"
    emb_dir = artifacts_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    model_name = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size = int(os.getenv("EMBED_BATCH", "64"))

    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset parquet not found: {ds_path}. Run: python -m ml.data.build_dataset")

    df = pd.read_parquet(ds_path)

    if "row_id" not in df.columns:
        raise ValueError("Dataset must contain row_id. Re-run: python -m ml.data.build_dataset")

    if "text" not in df.columns:
        raise ValueError("Dataset must contain text. Re-run: python -m ml.data.build_dataset")

    # IMPORTANT: row_ids order defines embedding order
    row_ids = df["row_id"].astype(str).tolist()
    texts = df["text"].fillna("").astype(str).tolist()

    emb = embed_texts(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        prefer_gpu=True,
        normalize=True,
    )

    if emb.shape[0] != len(row_ids):
        raise RuntimeError(f"Embedding rows mismatch: emb={emb.shape[0]} vs row_ids={len(row_ids)}")

    tag = _safe_model_tag(model_name)

    # Make outputs unique per model (prevents overwriting when you try multiple models)
    out_npy = str(emb_dir / f"emb_{tag}_float32.npy")
    out_index = str(emb_dir / f"emb_index_{tag}.jsonl")
    out_meta = str(emb_dir / f"embed_meta_{tag}.json")

    save_embeddings(
        row_ids=row_ids,
        embeddings=emb,
        out_npy=out_npy,
        out_index_jsonl=out_index,
        meta_path=out_meta,
        model_name=model_name,
    )

    # Also write a "latest pointer" file (training can read this)
    latest_ptr = emb_dir / "embed_meta.json"
    latest_ptr.write_text(json.dumps({
        "latest_meta": out_meta
    }, indent=2), encoding="utf-8")

    print("[ok] embeddings saved")
    print(" -", out_npy)
    print(" -", out_index)
    print(" -", out_meta)
    print(" - latest pointer:", latest_ptr)


if __name__ == "__main__":
    main()