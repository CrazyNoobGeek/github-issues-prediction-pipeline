from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .io_utils import ensure_dir


def load_json(path: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not os.path.exists(path):
        return dict(default or {})
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def repo_state_path(state_dir: str) -> str:
    return os.path.join(state_dir, "repos_state.json")


def issues_state_path(state_dir: str) -> str:
    return os.path.join(state_dir, "issues_state.json")


def get_since(issues_state: Dict[str, Any], repo_full_name: str, default_since: str) -> str:
    repos = issues_state.get("repos")
    if not isinstance(repos, dict):
        repos = {}
        issues_state["repos"] = repos
    entry = repos.get(repo_full_name) or {}
    if isinstance(entry, dict) and entry.get("since"):
        return str(entry["since"])
    return default_since


def set_since(issues_state: Dict[str, Any], repo_full_name: str, since_iso: str) -> None:
    repos = issues_state.get("repos")
    if not isinstance(repos, dict):
        repos = {}
        issues_state["repos"] = repos
    entry = repos.get(repo_full_name)
    if not isinstance(entry, dict):
        entry = {}
        repos[repo_full_name] = entry
    entry["since"] = since_iso


def set_repo_run_meta(
    issues_state: Dict[str, Any],
    repo_full_name: str,
    *,
    last_run_at: str,
    collected_count: int,
    truncated: bool,
    status: str,
) -> None:
    repos = issues_state.setdefault("repos", {})
    entry = repos.setdefault(repo_full_name, {})
    entry["last_run_at"] = last_run_at
    entry["last_collected_count"] = collected_count
    entry["last_truncated"] = truncated
    entry["last_status"] = status