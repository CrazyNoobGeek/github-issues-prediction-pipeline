# ml/inference/preprocess_single.py

from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np

from ml.data.clean_dedup import clean_and_dedup
from ml.features.build_tabular import add_features_and_targets
from ml.features.embed_text import embed_texts

TABULAR_COLS = [
    "comments", "num_assignees", "labels_count",
    "title_len", "body_len", "has_body",
    "created_dow", "created_hour",
    "ttf_hours",
]

def issue_to_features(
    issue: Dict[str, Any],
    embed_model_name: str,
    prefer_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame([issue])
    df = clean_and_dedup(df)
    if len(df) == 0:
        raise ValueError("Issue record is invalid after cleaning (missing repo_full_name/number/created_at).")
    df = add_features_and_targets(df)

    X_tab = df[TABULAR_COLS].fillna(0.0).astype(float).to_numpy()
    X_emb = embed_texts(df["text"].tolist(), model_name=embed_model_name, prefer_gpu=prefer_gpu)

    return X_tab, X_emb
