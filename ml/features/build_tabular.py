from __future__ import annotations
import pandas as pd


def add_features_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Text
    df["title"] = df["title"].fillna("")
    df["body"] = df["body"].fillna("")
    df["text"] = (df["title"] + " " + df["body"]).str.strip()

    # Labels
    df["labels_str"] = df["labels"].apply(lambda xs: " ".join(xs) if isinstance(xs, list) else "")
    df["labels_count"] = df["labels"].apply(lambda xs: len(xs) if isinstance(xs, list) else 0)

    # Lengths
    df["title_len"] = df["title"].str.len()
    df["body_len"] = df["body"].str.len()
    df["has_body"] = (df["body_len"] > 0).astype(int)

    # Time
    df["created_dow"] = df["created_ts"].dt.weekday
    df["created_hour"] = df["created_ts"].dt.hour

    df["ttf_hours"] = (df["first_comment_ts"] - df["created_ts"]).dt.total_seconds() / 3600.0
    df.loc[df["ttf_hours"] < 0, "ttf_hours"] = pd.NA

    # Targets
    df["days_to_close"] = (df["closed_ts"] - df["created_ts"]).dt.total_seconds() / 86400.0
    df["is_closed"] = df["closed_ts"].notna().astype(int)
    df["resolved_30d"] = ((df["days_to_close"].notna()) & (df["days_to_close"] <= 30.0)).astype(int)

    return df