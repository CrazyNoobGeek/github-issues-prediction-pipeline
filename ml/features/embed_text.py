# ml/features/embed_text.py

from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
import hashlib
import json  # ✅ FIX: missing


def get_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def embed_texts(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    prefer_gpu: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    device = get_device(prefer_gpu)
    model = SentenceTransformer(model_name, device=device)

    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return emb.astype(np.float32)


def cache_key(texts: List[str], model_name: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for t in texts[:2000]:
        h.update(t.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def embed_texts_cached(
    texts: List[str],
    cache_dir: str = "ml/models/cache",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    prefer_gpu: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = cache_key(texts, model_name)
    path = Path(cache_dir) / f"emb_{key}.npy"
    if path.exists():
        return np.load(path)
    emb = embed_texts(texts, model_name, batch_size, prefer_gpu, normalize)
    np.save(path, emb)
    return emb


def save_embeddings(
    row_ids: List[str],
    embeddings: np.ndarray,
    out_npy: str,
    out_index_jsonl: str,
    meta_path: str,
    model_name: str,
):
    Path(Path(out_npy).parent).mkdir(parents=True, exist_ok=True)

    embeddings = embeddings.astype(np.float32)
    np.save(out_npy, embeddings)

    # ✅ store mapping row_id -> idx
    with open(out_index_jsonl, "w", encoding="utf-8") as f:
        for i, rid in enumerate(row_ids):
            f.write(json.dumps({"row_id": rid, "idx": i}, ensure_ascii=False) + "\n")

    meta = {
        "model_name": model_name,
        "embeddings_shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "n_rows": len(row_ids),
        "npy_path": out_npy,
        "index_path": out_index_jsonl,
    }
    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
