from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Allow running as: cd ml && python data/build_dataset.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.data.load_jsonl import load_issues_from_jsonl
from ml.data.load_mongo import load_issues_from_mongo
from ml.data.clean_dedup import clean_and_dedup
from ml.features.build_tabular import add_features_and_targets


REQUIRED_RAW_COLS = [
    "repo_full_name", "number",
    "title", "body", "labels", "author_association",
    "comments", "num_assignees",
    "created_at", "updated_at", "closed_at", "first_comment_at",
    "is_pull_request",
]


def _resolve_jsonl_root(p: str) -> str:
    """Resolve a JSONL root path in a user-friendly way.

    Supports:
    - absolute paths
    - paths relative to repo root
    - paths relative to repo root/Data_Collector (common in this project)
    """
    raw = Path(p)
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                _REPO_ROOT / raw,
                _REPO_ROOT / "Data_Collector" / raw,
                _REPO_ROOT / "Data_collector" / raw,
                raw,  # relative to current working directory
            ]
        )

    for c in candidates:
        if c.exists() and c.is_dir():
            return str(c)

    # Nothing exists; return the original string (so error messages show what user passed)
    return p


def _default_jsonl_root() -> str:
    # Prefer the collector output path
    for c in [
        _REPO_ROOT / "Data_Collector" / "data" / "raw" / "issues",
        _REPO_ROOT / "Data_collector" / "data" / "raw" / "issues",
        _REPO_ROOT / "data" / "raw" / "issues",
    ]:
        if c.exists() and c.is_dir():
            return str(c)
    return "data/raw/issues"


def _read_config_simple() -> Dict[str, Any]:
    return {
        "data_source": os.getenv("DATA_SOURCE", "jsonl"),  # jsonl | mongo
        "jsonl_root": _resolve_jsonl_root(os.getenv("JSONL_ROOT", _default_jsonl_root())),
        "mongo_uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        "mongo_db": os.getenv("MONGO_DB", "github"),
        "mongo_collection": os.getenv("MONGO_COLLECTION", "issues_clean"),
        "artifacts_dir": os.getenv("ARTIFACTS_DIR", "ml/artifacts"),
        # optional: cap regression target stability
        "cap_days_to_close": float(os.getenv("CAP_DAYS_TO_CLOSE", "0")),  # 0 => disabled
    }


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in REQUIRED_RAW_COLS:
        if c not in df.columns:
            df[c] = None

    # Normalize a few types early
    # Ensure booleans are real booleans where possible
    if "is_pull_request" in df.columns:
        df["is_pull_request"] = df["is_pull_request"].fillna(False).astype(bool)

    return df


def _jsonable_ts(x: Any) -> Any:
    """Convert pandas Timestamp -> ISO string, keep None/NaT as None."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    # Timestamp -> ISO
    if isinstance(x, (pd.Timestamp,)):
        if x.tzinfo is None:
            x = x.tz_localize("UTC")
        return x.isoformat()
    # Already string or other
    return x


def main() -> None:
    cfg = _read_config_simple()
    artifacts_dir = Path(cfg["artifacts_dir"])
    ds_dir = artifacts_dir / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load raw records
    if cfg["data_source"].lower() == "mongo":
        records: List[Dict[str, Any]] = load_issues_from_mongo(
            cfg["mongo_uri"], cfg["mongo_db"], cfg["mongo_collection"]
        )
    else:
        records = load_issues_from_jsonl(cfg["jsonl_root"])

    if not records:
        jsonl_root = Path(cfg["jsonl_root"])
        n_files = len(list(jsonl_root.rglob("*.jsonl"))) if jsonl_root.exists() else 0
        raise RuntimeError(
            "No records loaded. "
            f"data_source={cfg['data_source']} jsonl_root={cfg['jsonl_root']} jsonl_files={n_files}. "
            "Your collector output is usually under Data_Collector/data/raw/issues. "
            "Try: JSONL_ROOT=Data_Collector/data/raw/issues"
        )

    df = pd.DataFrame(records)
    df = _ensure_required_columns(df)

    # 2) Clean + dedup (creates created_ts/updated_ts/closed_ts/first_comment_ts)
    df = clean_and_dedup(df)

    if len(df) == 0:
        raise RuntimeError("All rows dropped after clean_and_dedup(). Check input data.")

    # 3) Feature engineering + targets (creates title_len/body_len/has_body etc.)
    df = add_features_and_targets(df)

    # 4) Stable ID
    df["row_id"] = df["repo_full_name"].astype(str) + "#" + df["number"].astype(str)

    # Optional stability cap (mostly for regression training convenience)
    cap = float(cfg.get("cap_days_to_close", 0.0))
    if cap and cap > 0:
        # keep classification rows; cap only affects regression target distribution
        # You can either drop too-long closures or clip. Dropping is safer.
        df = df[(df["days_to_close"].isna()) | (df["days_to_close"] <= cap)].copy()

    # 5) Select columns for the final dataset (IMPORTANT: include engineered cols!)
    keep_cols = [
        # identifiers
        "row_id",
        "repo_full_name", "number",

        # raw
        "title", "body", "labels", "author_association",
        "comments", "num_assignees",
        "created_at", "updated_at", "closed_at", "first_comment_at",

        # cleaned timestamps
        "created_ts", "updated_ts", "closed_ts", "first_comment_ts",

        # engineered text + label cols
        "text", "labels_str", "labels_count",
        "title_len", "body_len", "has_body",

        # engineered time cols
        "created_dow", "created_hour", "ttf_hours",

        # targets
        "is_closed", "days_to_close", "resolved_30d",
    ]

    for c in keep_cols:
        if c not in df.columns:
            df[c] = None
    df = df[keep_cols].copy()

    # Ensure correct dtypes (helps parquet + later Spark)
    # timestamps already datetime in dataframe from clean_dedup
    for c in ["comments", "num_assignees", "labels_count", "title_len", "body_len", "has_body", "created_dow", "created_hour"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "ttf_hours" in df.columns:
        df["ttf_hours"] = pd.to_numeric(df["ttf_hours"], errors="coerce")

    # 6) Save parquet (recommended)
    out_parquet = ds_dir / "issues_clean_latest.parquet"
    df.to_parquet(out_parquet, index=False)

    # 7) Save one big JSONL (timestamps as ISO strings)
    out_jsonl = ds_dir / "issues_clean_latest.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            for k in ["created_ts", "updated_ts", "closed_ts", "first_comment_ts"]:
                rec[k] = _jsonable_ts(rec.get(k))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 8) Meta
    meta = {
        "n_rows": int(len(df)),
        "data_source": cfg["data_source"],
        "jsonl_root": cfg["jsonl_root"],
        "mongo_db": cfg["mongo_db"],
        "mongo_collection": cfg["mongo_collection"],
        "cap_days_to_close": cap,
        "parquet_path": str(out_parquet),
        "jsonl_path": str(out_jsonl),
        "columns": keep_cols,
    }
    meta_path = ds_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[ok] dataset built")
    print(" - parquet:", out_parquet)
    print(" - jsonl:  ", out_jsonl)
    print(" - meta:   ", meta_path)
    print(" - rows:   ", len(df))


if __name__ == "__main__":
    main()