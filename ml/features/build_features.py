import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional


def parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def build_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)

    # Drop PRs
    df = df[df["is_pull_request"] == False].copy()

    # Parse dates
    df["created_dt"] = df["created_at"].apply(parse_dt)
    df["closed_dt"] = df["closed_at"].apply(parse_dt)
    df["first_comment_dt"] = df["first_comment_at"].apply(parse_dt)

    df = df[df["created_dt"].notna()].copy()

    # Text
    df["title"] = df["title"].fillna("")
    df["body"] = df["body"].fillna("")
    df["text"] = (df["title"] + " " + df["body"]).str.strip()

    # Labels
    df["labels"] = df["labels"].apply(lambda x: x if isinstance(x, list) else [])
    df["labels_str"] = df["labels"].apply(lambda xs: " ".join(xs))
    df["labels_count"] = df["labels"].apply(len)

    # Numeric
    df["comments"] = df["comments"].fillna(0).astype(int)
    df["num_assignees"] = df["num_assignees"].fillna(0).astype(int)

    df["title_len"] = df["title"].str.len()
    df["body_len"] = df["body"].str.len()
    df["has_body"] = (df["body_len"] > 0).astype(int)

    # Time features
    df["created_dow"] = df["created_dt"].dt.weekday
    df["created_hour"] = df["created_dt"].dt.hour

    # Time to first response (hours)
    df["ttf_hours"] = (
        (df["first_comment_dt"] - df["created_dt"])
        .dt.total_seconds() / 3600
    )

    # Targets
    df["days_to_close"] = (
        (df["closed_dt"] - df["created_dt"])
        .dt.total_seconds() / 86400
    )

    df["resolved_30d"] = (
        df["days_to_close"].notna() & (df["days_to_close"] <= 30)
    ).astype(int)

    return df