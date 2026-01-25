from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .github_client import GitHubClient


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def keep_repo(repo: Dict[str, Any]) -> Dict[str, Any]:
    owner_login = None
    if isinstance(repo.get("owner"), dict):
        owner_login = repo["owner"].get("login")

    return {
        "collected_at": utc_now_iso(),
        "id": repo.get("id"),
        "node_id": repo.get("node_id"),
        "full_name": repo.get("full_name"),
        "name": repo.get("name"),
        "owner_login": owner_login,
        "html_url": repo.get("html_url"),
        "description": repo.get("description"),
        "created_at": repo.get("created_at"),
        "updated_at": repo.get("updated_at"),
        "pushed_at": repo.get("pushed_at"),
        "language": repo.get("language"),
        "topics": repo.get("topics", []) if isinstance(repo.get("topics"), list) else [],
        "stargazers_count": repo.get("stargazers_count"),
        "forks_count": repo.get("forks_count"),
        "watchers_count": repo.get("watchers_count"),
        "open_issues_count": repo.get("open_issues_count"),
        "archived": repo.get("archived"),
        "disabled": repo.get("disabled"),
        "has_issues": repo.get("has_issues"),
        "size": repo.get("size"),
        "license": (repo.get("license") or {}).get("spdx_id") if isinstance(repo.get("license"), dict) else None,
        "default_branch": repo.get("default_branch"),
        "visibility": repo.get("visibility"),
    }


def fetch_repos_from_org(gh: GitHubClient, org: str, max_pages: int = 20) -> List[Dict[str, Any]]:
    repos: List[Dict[str, Any]] = []
    for page in gh.paginate(
        f"/orgs/{org}/repos",
        params={"type": "public", "sort": "updated", "direction": "desc"},
        per_page=100,
        max_pages=max_pages,
    ):
        for r in page:
            if isinstance(r, dict):
                repos.append(keep_repo(r))
    return repos


def search_repos(gh: GitHubClient, query: str, max_pages: int = 10) -> List[Dict[str, Any]]:
    """
    /search/repositories returns a dict { items: [...] }
    We'll request page-by-page using normal get (not gh.paginate), because response is dict.
    """
    out: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        data = gh.get(
            "/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": 100, "page": page},
        )
        items = data.get("items", []) if isinstance(data, dict) else []
        if not items:
            break
        for r in items:
            if isinstance(r, dict):
                out.append(keep_repo(r))
    return out


def collect_repos(
    gh: GitHubClient,
    orgs: Optional[List[str]] = None,
    search_queries: Optional[List[str]] = None,
    max_repos_total: int = 200,
) -> List[Dict[str, Any]]:
    orgs = orgs or []
    search_queries = search_queries or []

    seen: Set[str] = set()
    output: List[Dict[str, Any]] = []

    for org in orgs:
        for r in fetch_repos_from_org(gh, org, max_pages=50):
            full = r.get("full_name")
            if not full or full in seen:
                continue
            seen.add(full)
            output.append(r)
            if len(output) >= max_repos_total:
                return output

    for q in search_queries:
        for r in search_repos(gh, q, max_pages=10):
            full = r.get("full_name")
            if not full or full in seen:
                continue
            seen.add(full)
            output.append(r)
            if len(output) >= max_repos_total:
                return output

    return output