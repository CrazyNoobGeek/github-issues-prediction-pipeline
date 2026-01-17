from typing import Any, Dict, List, Optional

from .github_client import GitHubClient


def keep_repo(repo: Dict[str, Any]) -> Dict[str, Any]:
    owner = (repo.get("owner") or {}).get("login") if isinstance(repo.get("owner"), dict) else None
    return {
        "collected_at": None,
        "id": repo.get("id"),
        "name": repo.get("name"),
        "full_name": repo.get("full_name"),
        "owner_login": owner,
        "html_url": repo.get("html_url"),
        "description": repo.get("description"),
        "language": repo.get("language"),
        "fork": repo.get("fork"),
        "stargazers_count": repo.get("stargazers_count"),
        "watchers_count": repo.get("watchers_count"),
        "forks_count": repo.get("forks_count"),
        "open_issues_count": repo.get("open_issues_count"),
        "created_at": repo.get("created_at"),
        "updated_at": repo.get("updated_at"),
        "pushed_at": repo.get("pushed_at"),
        "license": (repo.get("license") or {}).get("spdx_id") if isinstance(repo.get("license"), dict) else None,
        "topics": repo.get("topics") if isinstance(repo.get("topics"), list) else [],
    }


def _collect_org_repos(gh: GitHubClient, org: str, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
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


def _collect_search_repos(gh: GitHubClient, query: str, max_pages: int = 10) -> List[Dict[str, Any]]:
    # GitHub search API caps results; keep paging until empty or max_pages.
    kept: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        data = gh.get(
            "/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": 100, "page": page},
        )
        items = (data or {}).get("items") if isinstance(data, dict) else None
        if not items:
            break
        for r in items:
            if isinstance(r, dict):
                kept.append(keep_repo(r))
    return kept


def collect_repos(
    gh: GitHubClient,
    orgs: Optional[List[str]] = None,
    search_queries: Optional[List[str]] = None,
    max_repos_total: int = 200,
) -> List[Dict[str, Any]]:
    orgs = orgs or []
    search_queries = search_queries or []

    by_full_name: Dict[str, Dict[str, Any]] = {}

    for org in orgs:
        for r in _collect_org_repos(gh, org):
            fn = r.get("full_name")
            if fn:
                by_full_name[fn] = r
            if len(by_full_name) >= max_repos_total:
                return list(by_full_name.values())[:max_repos_total]

    for q in search_queries:
        for r in _collect_search_repos(gh, q):
            fn = r.get("full_name")
            if fn:
                by_full_name.setdefault(fn, r)
            if len(by_full_name) >= max_repos_total:
                return list(by_full_name.values())[:max_repos_total]

    return list(by_full_name.values())[:max_repos_total]

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .github_client import GitHubClient


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def keep_repo(repo: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "collected_at": utc_now_iso(),
        "id": repo.get("id"),
        "node_id": repo.get("node_id"),
        "full_name": repo.get("full_name"),
        "name": repo.get("name"),
        "owner": (repo.get("owner") or {}).get("login"),
        "html_url": repo.get("html_url"),
        "description": repo.get("description"),
        "created_at": repo.get("created_at"),
        "updated_at": repo.get("updated_at"),
        "pushed_at": repo.get("pushed_at"),
        "language": repo.get("language"),
        "topics": repo.get("topics", []),
        "stargazers_count": repo.get("stargazers_count"),
        "forks_count": repo.get("forks_count"),
        "watchers_count": repo.get("watchers_count"),
        "open_issues_count": repo.get("open_issues_count"),
        "archived": repo.get("archived"),
        "disabled": repo.get("disabled"),
        "license": (repo.get("license") or {}).get("spdx_id"),
        "default_branch": repo.get("default_branch"),
    }


def fetch_repos_from_org(gh: GitHubClient, org: str, max_pages: int = 20) -> List[Dict[str, Any]]:
    repos = []
    for page in gh.paginate(f"/orgs/{org}/repos",
                            params={"type": "public", "sort": "updated", "direction": "desc"},
                            per_page=100, max_pages=max_pages):
        repos.extend(page)
    return repos


def search_repos(gh: GitHubClient, query: str, max_pages: int = 10) -> List[Dict[str, Any]]:
    """
    /search/repositories returns dict: { items: [...] }
    We'll paginate with page.
    """
    all_items = []
    for page in gh.paginate("/search/repositories",
                            params={"q": query, "sort": "stars", "order": "desc"},
                            per_page=100, max_pages=max_pages):
        # each "page" is [dict] because paginate wraps dict as one-page list
        d = page[0]
        items = d.get("items", [])
        all_items.extend(items)
    return all_items


def collect_repos(gh: GitHubClient,
                  orgs: List[str],
                  search_queries: List[str],
                  max_repos_total: int = 200) -> List[Dict[str, Any]]:
    """
    Collect repos from multiple sources, deduplicate by full_name, keep up to max_repos_total.
    """
    seen: Set[str] = set()
    output: List[Dict[str, Any]] = []

    # 1) Orgs
    for org in orgs:
        repos = fetch_repos_from_org(gh, org, max_pages=50)
        for r in repos:
            full = r.get("full_name")
            if not full or full in seen:
                continue
            seen.add(full)
            output.append(keep_repo(r))
            if len(output) >= max_repos_total:
                return output

    # 2) Search queries
    for q in search_queries:
        repos = search_repos(gh, q, max_pages=10)
        for r in repos:
            full = r.get("full_name")
            if not full or full in seen:
                continue
            seen.add(full)
            output.append(keep_repo(r))
            if len(output) >= max_repos_total:
                return output

    return output