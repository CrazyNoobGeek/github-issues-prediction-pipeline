from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dtparser

from .github_client import GitHubClient


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_isoparse(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return dtparser.isoparse(s)
    except Exception:
        return None


def subtract_overlap(iso_ts: str, overlap_seconds: int) -> str:
    """
    Reduce cursor time a bit to avoid missing boundary updates.
    """
    d = _safe_isoparse(iso_ts)
    if not d:
        return iso_ts
    d2 = d - timedelta(seconds=max(0, overlap_seconds))
    # Keep Z format if you prefer; ISO with timezone is fine too.
    return d2.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def keep_issue(issue: Dict[str, Any], repo_full_name: str) -> Dict[str, Any]:
    user = issue.get("user") or {}
    assignees = issue.get("assignees") or []
    labels = issue.get("labels") or []

    label_names: List[str] = []
    for lb in labels:
        if isinstance(lb, dict):
            name = lb.get("name")
            if name:
                label_names.append(str(name))
        else:
            label_names.append(str(lb))

    # /issues endpoint includes PRs; mark them to filter later
    is_pr = bool(issue.get("pull_request"))

    return {
        "collected_at": utc_now_iso(),
        "repo_full_name": repo_full_name,

        # Keys
        "id": issue.get("id"),
        "node_id": issue.get("node_id"),
        "number": issue.get("number"),

        # Resolution signals (targets)
        "state": issue.get("state"),                 # open/closed
        "state_reason": issue.get("state_reason"),   # completed/not_planned/reopened? (varies)

        # Text (features)
        "title": issue.get("title"),
        "body": issue.get("body"),

        # User metadata (features)
        "user_login": user.get("login"),
        "author_association": issue.get("author_association"),

        # Labels / activity (features)
        "labels": label_names,
        "comments": issue.get("comments"),

        # Time (targets + features)
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "closed_at": issue.get("closed_at"),

        # Assignment / misc
        "assignees": [a.get("login") for a in assignees if isinstance(a, dict) and a.get("login")],
        "num_assignees": len([a for a in assignees if isinstance(a, dict)]),
        "locked": issue.get("locked"),
        "active_lock_reason": issue.get("active_lock_reason"),
        "milestone": (issue.get("milestone") or {}).get("title") if issue.get("milestone") else None,

        # URLs (debug / trace)
        "html_url": issue.get("html_url"),
        "url": issue.get("url"),

        "is_pull_request": is_pr,

        # Will be filled optionally
        "first_comment_at": None,   # strong feature: time-to-first-response
    }


def max_updated_at(kept: List[Dict[str, Any]]) -> Optional[str]:
    max_dt: Optional[datetime] = None
    max_iso: Optional[str] = None
    for r in kept:
        ua = r.get("updated_at")
        d = _safe_isoparse(ua)
        if not d:
            continue
        if max_dt is None or d > max_dt:
            max_dt = d
            max_iso = ua
    return max_iso


def fetch_first_comment_at(
    gh: GitHubClient,
    repo_full_name: str,
    issue_number: int,
) -> Optional[str]:
    """
    Fetch timestamp of the very first comment for an issue.
    Minimal cost: per_page=1, direction=asc.
    """
    owner, repo = repo_full_name.split("/", 1)
    data = gh.get(
        f"/repos/{owner}/{repo}/issues/{issue_number}/comments",
        params={"per_page": 1, "page": 1, "sort": "created", "direction": "asc"},
    )
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return first.get("created_at")
    return None


def collect_issues_for_repo(
    gh: GitHubClient,
    repo_full_name: str,
    since_iso: str,
    include_state: str = "all",
    max_pages: int = 50,
    collect_first_comment: bool = True,
    max_first_comment_fetch: int = 400,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Collect issues updated since `since_iso`.
    Returns (kept_issues, truncated_flag).
    """
    owner, repo = repo_full_name.split("/", 1)

    params = {
        "state": include_state,   # all/open/closed
        "since": since_iso,
        "sort": "updated",
        "direction": "asc",
    }

    kept: List[Dict[str, Any]] = []
    pages_seen = 0

    for page in gh.paginate(
        f"/repos/{owner}/{repo}/issues",
        params=params,
        per_page=100,
        max_pages=max_pages,
    ):
        pages_seen += 1
        for it in page:
            if not isinstance(it, dict):
                continue
            kept.append(keep_issue(it, repo_full_name))

    truncated = (pages_seen >= max_pages)

    # Optional: first comment timestamp (very useful ML feature)
    if collect_first_comment and kept:
        fetched = 0
        for r in kept:
            # Skip PRs and issues with 0 comments (no first comment exists)
            if r.get("is_pull_request"):
                continue
            if (r.get("comments") or 0) <= 0:
                continue
            if fetched >= max_first_comment_fetch:
                break

            num = r.get("number")
            if not isinstance(num, int):
                continue

            r["first_comment_at"] = fetch_first_comment_at(gh, repo_full_name, num)
            fetched += 1

    return kept, truncated