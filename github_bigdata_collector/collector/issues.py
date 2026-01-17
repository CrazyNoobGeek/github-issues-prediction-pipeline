from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dateutil import parser as dtparser

from .github_client import GitHubClient


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def keep_issue(issue: Dict[str, Any], repo_full_name: str) -> Dict[str, Any]:
    user = issue.get("user") or {}
    assignees = issue.get("assignees") or []
    labels = issue.get("labels") or []

    label_names = []
    for lb in labels:
        if isinstance(lb, dict):
            label_names.append(lb.get("name"))
        else:
            label_names.append(str(lb))

    return {
        "collected_at": utc_now_iso(),
        "repo_full_name": repo_full_name,

        "id": issue.get("id"),
        "node_id": issue.get("node_id"),
        "number": issue.get("number"),

        "state": issue.get("state"),
        "state_reason": issue.get("state_reason"),

        "title": issue.get("title"),
        "body": issue.get("body"),

        "user_login": user.get("login"),
        "author_association": issue.get("author_association"),

        "labels": [x for x in label_names if x],
        "comments": issue.get("comments"),

        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "closed_at": issue.get("closed_at"),

        "assignees": [a.get("login") for a in assignees if isinstance(a, dict)],
        "num_assignees": len(assignees),

        "locked": issue.get("locked"),
        "active_lock_reason": issue.get("active_lock_reason"),

        "milestone": (issue.get("milestone") or {}).get("title") if issue.get("milestone") else None,

        "html_url": issue.get("html_url"),
        "url": issue.get("url"),

        # IMPORTANT: /issues endpoint includes PRs; mark them to filter later in Spark cleaning
        "is_pull_request": bool(issue.get("pull_request")),
    }


def max_updated_at(kept: List[Dict[str, Any]]) -> Optional[str]:
    max_dt = None
    max_iso = None
    for r in kept:
        ua = r.get("updated_at")
        if not ua:
            continue
        try:
            d = dtparser.isoparse(ua)
            if (max_dt is None) or (d > max_dt):
                max_dt = d
                max_iso = ua
        except Exception:
            continue
    return max_iso


def collect_issues_for_repo(
    gh: GitHubClient,
    repo_full_name: str,
    since_iso: str,
    include_state: str = "all",
    max_pages: int = 50
) -> List[Dict[str, Any]]:
    owner, repo = repo_full_name.split("/", 1)
    params = {
        "state": include_state,  # all/open/closed
        "since": since_iso,
        "sort": "updated",
        "direction": "asc",
    }

    kept: List[Dict[str, Any]] = []
    for page in gh.paginate(f"/repos/{owner}/{repo}/issues",
                            params=params, per_page=100, max_pages=max_pages):
        for it in page:
            kept.append(keep_issue(it, repo_full_name))

    return kept