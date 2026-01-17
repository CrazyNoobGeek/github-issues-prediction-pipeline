import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import json
from datetime import datetime
from typing import Dict, Any, List

from tqdm import tqdm

from github_bigdata_collector.collector.github_client import GitHubClient
from github_bigdata_collector.collector.io_utils import write_jsonl, ensure_dir
from github_bigdata_collector.collector.state import (
    load_json, save_json,
    repo_state_path, issues_state_path,
    get_since, set_since
)
from github_bigdata_collector.collector.repos import collect_repos
from github_bigdata_collector.collector.issues import collect_issues_for_repo, max_updated_at


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def today_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d")


def main():
    config_path = os.getenv("COLLECTOR_CONFIG", "config.json")
    cfg = load_config(config_path)

    raw_dir = cfg["raw_dir"]
    state_dir = cfg["state_dir"]
    log_dir = cfg.get("log_dir", "logs")

    ensure_dir(raw_dir)
    ensure_dir(state_dir)
    ensure_dir(log_dir)

    token = os.getenv("GITHUB_TOKEN") or cfg.get("github_token")
    if not token:
        print("[auth] Missing GitHub token.")
        print("[auth] Set the environment variable GITHUB_TOKEN and re-run.")
        print("[auth] PowerShell example:")
        print("        $env:GITHUB_TOKEN = 'YOUR_TOKEN_HERE'")
        print("        C:/Users/z.bougayou/AppData/Local/miniconda3/envs/github-bigdata/python.exe run_collector.py")
        raise SystemExit(1)

    gh = GitHubClient(token=token)

    # --- Load states
    repos_state_file = repo_state_path(state_dir)
    issues_state_file = issues_state_path(state_dir)

    repos_state = load_json(repos_state_file, default={"last_refresh": None})
    issues_state = load_json(issues_state_file, default={"repos": {}})

    # --- 1) Collect repos (refresh list)
    orgs = cfg["repo_sources"].get("orgs", [])
    search_queries = cfg["repo_sources"].get("search_queries", [])
    max_repos_total = int(cfg["repo_sources"].get("max_repos_total", 200))

    repos = collect_repos(gh, orgs=orgs, search_queries=search_queries, max_repos_total=max_repos_total)

    repos_out_dir = os.path.join(raw_dir, "repos")
    ensure_dir(repos_out_dir)

    dated_repos_path = os.path.join(repos_out_dir, f"repos_{today_stamp()}.jsonl")
    latest_repos_path = os.path.join(repos_out_dir, "repos_latest.jsonl")

    # Write dated snapshot + overwrite latest
    write_jsonl(dated_repos_path, repos)
    # overwrite latest
    with open(latest_repos_path, "w", encoding="utf-8") as f:
        for r in repos:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    repos_state["last_refresh"] = datetime.utcnow().isoformat() + "Z"
    save_json(repos_state_file, repos_state)

    repo_full_names = [r["full_name"] for r in repos if r.get("full_name")]

    print(f"[repos] collected {len(repo_full_names)} repos")
    print(f"[repos] snapshot: {dated_repos_path}")
    print(f"[repos] latest:   {latest_repos_path}")

    # --- 2) Collect issues incrementally
    default_since = cfg["issues_collection"].get("default_since", "2024-01-01T00:00:00Z")
    max_repos_per_run = int(cfg["issues_collection"].get("max_repos_per_run", len(repo_full_names)))
    max_pages_per_repo = int(cfg["issues_collection"].get("max_pages_per_repo", 50))
    include_state = cfg["issues_collection"].get("include_state", "all")

    repo_full_names = repo_full_names[:max_repos_per_run]

    issues_base_dir = os.path.join(raw_dir, "issues")
    ensure_dir(issues_base_dir)

    total_written = 0

    for full in tqdm(repo_full_names, desc="Collecting issues"):
        since = get_since(issues_state, full, default_since)

        try:
            issues = collect_issues_for_repo(
                gh,
                repo_full_name=full,
                since_iso=since,
                include_state=include_state,
                max_pages=max_pages_per_repo
            )
        except Exception as e:
            print(f"[issues] ERROR repo={full}: {e}")
            continue

        if not issues:
            continue

        owner, repo = full.split("/", 1)
        out_path = os.path.join(issues_base_dir, owner, f"{repo}.jsonl")
        write_jsonl(out_path, issues)
        total_written += len(issues)

        mx = max_updated_at(issues)
        if mx:
            set_since(issues_state, full, mx)

    save_json(issues_state_file, issues_state)

    print(f"[issues] wrote {total_written} issue records (raw JSONL)")
    print(f"[state] saved: {issues_state_file}")
    print("[done] collector run complete")


if __name__ == "__main__":
    main()