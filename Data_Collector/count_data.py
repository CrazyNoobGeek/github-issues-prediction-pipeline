import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.json"


def resolve_raw_dir(raw_dir: str | None, config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    if raw_dir:
        candidate = Path(raw_dir)
        return candidate if candidate.is_absolute() else (BASE_DIR / candidate)

    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            cfg_raw = cfg.get("raw_dir")
            if isinstance(cfg_raw, str) and cfg_raw.strip():
                candidate = Path(cfg_raw)
                return candidate if candidate.is_absolute() else (BASE_DIR / candidate)
        except Exception:
            pass

    return BASE_DIR / "data/raw"


def count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def extract_repo_from_path(path: Path) -> str | None:
    """
    Supports:
      1) data/raw/issues/<org>/<repo>.jsonl  -> org/repo
      2) data/raw/issues/issues_<owner__repo>_<ts>.jsonl -> owner/repo
    """
    parts = path.parts

    # Case 1: .../issues/<org>/<repo>.jsonl
    try:
        issues_idx = parts.index("issues")
        # needs at least issues/org/file.jsonl
        if len(parts) >= issues_idx + 3:
            org = parts[issues_idx + 1]
            file = parts[issues_idx + 2]
            if file.endswith(".jsonl") and not file.startswith("issues_"):
                repo = file[:-5]
                return f"{org}/{repo}"
    except ValueError:
        pass

    # Case 2: issues_foo__bar_YYYYMMDD_HHMMSS.jsonl
    name = path.name
    m = re.match(r"issues_([A-Za-z0-9_.-]+)__([A-Za-z0-9_.-]+)_\d{8}_\d{6}\.jsonl$", name)
    if m:
        owner, repo = m.group(1), m.group(2)
        return f"{owner}/{repo}"

    return None


def extract_org(repo_full: str | None) -> str | None:
    if not repo_full or "/" not in repo_full:
        return None
    return repo_full.split("/", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize collected GitHub JSONL data")
    parser.add_argument(
        "--raw-dir",
        default=os.environ.get("RAW_DIR"),
        help="Path to raw data dir (defaults to config.json raw_dir, else Data_Collector/data/raw).",
    )
    args = parser.parse_args()

    raw_dir = resolve_raw_dir(args.raw_dir)
    issues_dir = raw_dir / "issues"
    repos_dir = raw_dir / "repos"

    print("RAW_DIR:", str(raw_dir))
    print("RAW_DIR exists:", raw_dir.exists())
    if raw_dir.exists():
        print("RAW_DIR content:", [p.name for p in raw_dir.iterdir()])


    repo_total_records = 0
    repo_files = []

    if repos_dir.exists():
        for p in repos_dir.rglob("*.jsonl"):
            repo_files.append(p)
            repo_total_records += count_lines(p)


    total_issues = 0
    issue_files = []
    org_issue_counts = defaultdict(int)
    repo_issue_counts = defaultdict(int)

    if issues_dir.exists():
        for p in issues_dir.rglob("*.jsonl"):
            issue_files.append(p)
            n = count_lines(p)
            total_issues += n

            repo_full = extract_repo_from_path(p)
            if repo_full:
                repo_issue_counts[repo_full] += n
                org = extract_org(repo_full)
                if org:
                    org_issue_counts[org] += n


    print("\n================ DATA SUMMARY ================\n")

    print(f"📦 Repo JSONL files found      : {len(repo_files)}")
    print(f"📦 Repo records (all files)    : {repo_total_records}")

    print(f"📄 Issue JSONL files found     : {len(issue_files)}")
    print(f"🐞 Issue records (all files)   : {total_issues}")

    if org_issue_counts:
        print("\n--- Issues per organization (best effort) ---")
        for org, c in sorted(org_issue_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {org:<20} : {c}")
    else:
        print("\n--- Issues per organization ---")
        print("  (Could not infer org/repo from filenames. Your issues files may be in a different format.)")

    if repo_issue_counts:
        print("\n--- Top 10 repositories by issue count (best effort) ---")
        for repo, c in sorted(repo_issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {repo:<40} : {c}")
    else:
        print("\n--- Top 10 repositories by issue count ---")
        print("  (Could not infer repo names from filenames.)")

    print("\n==============================================\n")


    if repo_files:
        print("[debug] example repo file:", repo_files[0])
    if issue_files:
        print("[debug] example issue file:", issue_files[0])


if __name__ == "__main__":
    main()