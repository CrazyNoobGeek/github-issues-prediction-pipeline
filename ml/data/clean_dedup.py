# ml/data/clean_dedup.py

from __future__ import annotations
from typing import Optional
import pandas as pd


def _to_ts(s: Optional[str]) -> pd.Timestamp:
    if not s:
        return pd.NaT
    return pd.to_datetime(s, utc=True, errors="coerce")


def clean_and_dedup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in [
        "repo_full_name", "number", "title", "body", "labels", "author_association",
        "comments", "num_assignees",
        "created_at", "updated_at", "closed_at", "first_comment_at",
        "is_pull_request",
    ]:
        if col not in df.columns:
            df[col] = None

    # Drop PRs
    df = df[df["is_pull_request"] == False].copy()

    # Normalize labels
    def norm_labels(x):
        if isinstance(x, list):
            return [str(t) for t in x if t is not None]
        return []
    df["labels"] = df["labels"].apply(norm_labels)

    # Cast numeric
    df["comments"] = pd.to_numeric(df["comments"], errors="coerce").fillna(0).astype(int)
    df["num_assignees"] = pd.to_numeric(df["num_assignees"], errors="coerce").fillna(0).astype(int)
    df["number"] = pd.to_numeric(df["number"], errors="coerce").astype("Int64")

    # Timestamps
    df["created_ts"] = df["created_at"].apply(_to_ts)
    df["updated_ts"] = df["updated_at"].apply(_to_ts)
    df["closed_ts"] = df["closed_at"].apply(_to_ts)
    df["first_comment_ts"] = df["first_comment_at"].apply(_to_ts)

    # Keep required identifiers
    df = df[df["created_ts"].notna()].copy()
    df = df[df["repo_full_name"].notna()].copy()
    df = df[df["number"].notna()].copy()

    # Remove exact duplicate snapshots (same repo/number and same updated_ts)
    df = df.sort_values(["repo_full_name", "number", "updated_ts"]).drop_duplicates(
        subset=["repo_full_name", "number", "updated_ts"], keep="last"
    )

    # Dedup to latest issue state
    df["dedup_ts"] = df["updated_ts"].fillna(df["created_ts"])
    df = df.sort_values(["repo_full_name", "number", "dedup_ts"]).drop_duplicates(
        subset=["repo_full_name", "number"], keep="last"
    )

    # Sanity: closed before created -> set closed to NaT
    bad_close = df["closed_ts"].notna() & (df["closed_ts"] < df["created_ts"])
    df.loc[bad_close, "closed_ts"] = pd.NaT

    # Sanity: first comment before created -> set to NaT
    bad_fc = df["first_comment_ts"].notna() & (df["first_comment_ts"] < df["created_ts"])
    df.loc[bad_fc, "first_comment_ts"] = pd.NaT

    return df.reset_index(drop=True)