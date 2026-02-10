from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from github_bigdata_pipeline.collector.github_client import GitHubClient
from github_bigdata_pipeline.collector.io_utils import ensure_dir, write_jsonl
from github_bigdata_pipeline.collector.repos import collect_repos
from github_bigdata_pipeline.collector.issues import collect_issues_for_repo, max_updated_at, subtract_overlap
from github_bigdata_pipeline.collector.state import (
    load_json,
    save_json,
    repo_state_path,
    issues_state_path,
    get_since,
    set_since,
    set_repo_run_meta,
)
from github_collector.collector.github_client import GitHubClient
from github_collector.collector.io_utils import ensure_dir, write_jsonl
from github_collector.collector.repos import collect_repos
from github_collector.collector.issues import collect_issues_for_repo, max_updated_at, subtract_overlap


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize_repo_full_name(repo_full_name: str) -> str:
    # owner/repo -> owner__repo
    s = repo_full_name.replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    cfg_path = os.getenv("COLLECTOR_CONFIG", "config.json")
    cfg = load_config(cfg_path)

    raw_dir = cfg.get("raw_dir", "data/raw")
    state_dir = cfg.get("state_dir", "data/state")
    log_dir = cfg.get("log_dir", "logs")

    ensure_dir(raw_dir)
    ensure_dir(state_dir)
    ensure_dir(log_dir)

    gh = GitHubClient()

    repos_state_file = repo_state_path(state_dir)
    issues_state_file = issues_state_path(state_dir)

    repos_state = load_json(repos_state_file, default={})
    issues_state = load_json(issues_state_file, default={})

    # --- Collect repositories
    repo_sources = cfg.get("repo_sources", {}) or {}
    orgs = repo_sources.get("orgs", []) or []
    search_queries = repo_sources.get("search_queries", []) or []
    max_repos_total = int(repo_sources.get("max_repos_total", 200))

    repos: List[Dict[str, Any]] = collect_repos(
        gh,
        orgs=orgs,
        search_queries=search_queries,
        max_repos_total=max_repos_total,
    )

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    repos_out = os.path.join(raw_dir, "repos", f"repos_{run_ts}.jsonl")
    n_repos_written = write_jsonl(repos_out, repos, mode="a")
    print(f"[repos] collected={len(repos)} written={n_repos_written} -> {repos_out}")

    # Save basic repo run info
    repos_state["last_run_at"] = utc_now_iso()
    repos_state["last_repos_count"] = len(repos)
    save_json(repos_state_file, repos_state)

    # --- Collect issues per repo incrementally
    icfg = cfg.get("issues_collection", {}) or {}
    default_since = str(icfg.get("default_since", "2024-01-01T00:00:00Z"))
    max_repos_per_run = int(icfg.get("max_repos_per_run", 200))
    max_pages_per_repo = int(icfg.get("max_pages_per_repo", 50))
    include_state = str(icfg.get("include_state", "all"))
    overlap_seconds = int(icfg.get("since_overlap_seconds", 600))  # 10 minutes default
    collect_first_comment = bool(icfg.get("collect_first_comment", True))
    max_first_comment_fetch = int(icfg.get("max_first_comment_fetch", 400))

    total_issues = 0
    truncated_repos = 0
    errors = 0


    repos_to_process = repos[:max_repos_per_run]

    for idx, r in enumerate(repos_to_process, start=1):
        repo_full = r.get("full_name")
        if not repo_full:
            continue

        # If repo disables issues, skip
        if r.get("has_issues") is False:
            print(f"[{idx}/{len(repos_to_process)}] [skip] {repo_full} has_issues=False")
            continue

        since = get_since(issues_state, repo_full, default_since)
        print(f"[{idx}/{len(repos_to_process)}] [issues] {repo_full} since={since}")

        try:
            kept, truncated = collect_issues_for_repo(
                gh,
                repo_full_name=repo_full,
                since_iso=since,
                include_state=include_state,
                max_pages=max_pages_per_repo,
                collect_first_comment=collect_first_comment,
                max_first_comment_fetch=max_first_comment_fetch,
            )

            # Write results (append)
            issues_out = os.path.join(
                raw_dir,
                "issues",
                f"issues_{sanitize_repo_full_name(repo_full)}_{run_ts}.jsonl",
            )
            n_written = write_jsonl(issues_out, kept, mode="a")
            total_issues += n_written

            # Update cursor
            mx = max_updated_at(kept)
            if mx:
                new_since = subtract_overlap(mx, overlap_seconds)
                set_since(issues_state, repo_full, new_since)

            set_repo_run_meta(
                issues_state,
                repo_full,
                last_run_at=utc_now_iso(),
                collected_count=n_written,
                truncated=bool(truncated),
                status="ok",
            )

            if truncated:
                truncated_repos += 1
                print(f"  [warn] repo truncated (max_pages reached): {repo_full}")

            print(f"  [ok] issues_collected={n_written} -> {issues_out}")

        except Exception as e:
            errors += 1
            set_repo_run_meta(
                issues_state,
                repo_full,
                last_run_at=utc_now_iso(),
                collected_count=0,
                truncated=False,
                status=f"error: {type(e).__name__}",
            )
            print(f"  [error] {repo_full}: {e}")

        # Save state after each repo to be crash-safe
        save_json(issues_state_file, issues_state)

    print(
        f"[done] repos={len(repos_to_process)} issues={total_issues} "
        f"truncated_repos={truncated_repos} errors={errors}"
    )


if __name__ == "__main__":
    main()