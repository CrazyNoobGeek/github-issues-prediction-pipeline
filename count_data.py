import os
import json
from collections import defaultdict


RAW_DIR = "data/raw"


def count_jsonl_lines(path: str) -> int:
    """Count number of lines in a JSONL file safely."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def main():
    issues_dir = os.path.join(RAW_DIR, "issues")
    repos_dir = os.path.join(RAW_DIR, "repos")
    print("RAW_DIR exists:", os.path.exists(RAW_DIR))
    print("RAW_DIR content:", os.listdir(RAW_DIR) if os.path.exists(RAW_DIR) else "NOT FOUND")


    # ---------- Repositories ----------
    repos_latest = os.path.join(repos_dir, "repos_latest.jsonl")
    repo_count = 0

    if os.path.exists(repos_latest):
        repo_count = count_jsonl_lines(repos_latest)

    # ---------- Issues ----------
    total_issues = 0
    org_issue_counts = defaultdict(int)
    repo_issue_counts = defaultdict(int)
    repo_file_count = 0

    if os.path.exists(issues_dir):
        for org in os.listdir(issues_dir):
            org_path = os.path.join(issues_dir, org)
            if not os.path.isdir(org_path):
                continue

            for file in os.listdir(org_path):
                if not file.endswith(".jsonl"):
                    continue

                repo_file_count += 1
                file_path = os.path.join(org_path, file)
                n = count_jsonl_lines(file_path)

                total_issues += n
                org_issue_counts[org] += n
                repo_name = f"{org}/{file.replace('.jsonl', '')}"
                repo_issue_counts[repo_name] += n

    # ---------- PRINT REPORT ----------
    print("\n================ DATA SUMMARY ================\n")

    print(f"📦 Repositories collected     : {repo_count}")
    print(f"📄 Repository issue files     : {repo_file_count}")
    print(f"🐞 Total issues collected     : {total_issues}")

    print("\n--- Issues per organization ---")
    for org, count in sorted(org_issue_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {org:<15} : {count}")

    print("\n--- Top 10 repositories by issue count ---")
    for repo, count in sorted(repo_issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {repo:<30} : {count}")

    print("\n==============================================\n")


if __name__ == "__main__":
    main()