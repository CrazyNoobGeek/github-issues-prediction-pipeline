#!/usr/bin/env python3
"""Merge issue dumps into a single JSON array.

Default behavior (matches repo layout):
- Reads:   Data_Collector/data/raw/issues/*.(json|jsonl|ndjson)
- Writes:  Data_Collector/issues_merged.json

The script is defensive:
- Supports `.json` (object or list) and `.jsonl`/`.ndjson` (one object per line).
- Streams output to avoid holding everything in memory.
- By default, skips malformed JSONL lines and unreadable files and writes a report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _write_one(out_fp, record: Any, *, first_written: bool) -> bool:
    if first_written:
        out_fp.write(",\n")
    out_fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    return True


def _write_records_from_file(
    path: Path,
    out_fp,
    *,
    first_written: bool,
    strict: bool,
) -> tuple[bool, int, int]:
    """Append records from `path` into `out_fp`.

    Returns: (first_written, records_written, bad_lines)
    """
    records_written = 0
    bad_lines = 0

    # Try regular JSON first.
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError:
        data = None

    if data is not None:
        if isinstance(data, list):
            for record in data:
                first_written = _write_one(out_fp, record, first_written=first_written)
                records_written += 1
        else:
            first_written = _write_one(out_fp, data, first_written=first_written)
            records_written += 1
        return first_written, records_written, bad_lines

    # Fallback: JSONL/NDJSON
    with path.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                if strict:
                    raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {e}") from e
                bad_lines += 1
                continue
            first_written = _write_one(out_fp, record, first_written=first_written)
            records_written += 1

    return first_written, records_written, bad_lines


def merge_issues(
    input_dir: Path,
    output_file: Path,
    *,
    strict: bool,
    report_file: Path | None,
) -> tuple[int, int, int, int]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    files_set = set()
    for pattern in ("*.json", "*.jsonl", "*.ndjson"):
        files_set.update(input_dir.glob(pattern))
    files = sorted(files_set)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_files = len(files)
    processed_files = 0
    skipped_files = 0
    bad_lines_total = 0
    total_records = 0
    problems: list[str] = []

    with output_file.open("w", encoding="utf-8") as out_fp:
        out_fp.write("[\n")
        first_written = False

        for file_path in files:
            try:
                first_written, written, bad_lines = _write_records_from_file(
                    file_path,
                    out_fp,
                    first_written=first_written,
                    strict=strict,
                )
            except Exception as e:  # noqa: BLE001 (CLI tool; we want to keep going in non-strict mode)
                if strict:
                    raise
                skipped_files += 1
                problems.append(f"SKIP {file_path}: {e}")
                continue

            processed_files += 1
            total_records += written
            bad_lines_total += bad_lines
            if bad_lines:
                problems.append(f"WARN {file_path}: skipped {bad_lines} malformed JSONL lines")

        out_fp.write("\n]\n")

    if report_file is not None:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with report_file.open("w", encoding="utf-8") as rep:
            rep.write(f"Input dir: {input_dir}\n")
            rep.write(f"Output: {output_file}\n")
            rep.write(f"Total files matched: {total_files}\n")
            rep.write(f"Processed files: {processed_files}\n")
            rep.write(f"Skipped files: {skipped_files}\n")
            rep.write(f"Total records written: {total_records}\n")
            rep.write(f"Malformed JSONL lines skipped: {bad_lines_total}\n")
            if problems:
                rep.write("\nDetails:\n")
                rep.write("\n".join(problems))
                rep.write("\n")

    return total_files, processed_files, total_records, bad_lines_total


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "Data_Collector" / "data" / "raw" / "issues"
    default_output = repo_root / "Data_Collector" / "issues_merged.json"
    default_report = repo_root / "Data_Collector" / "issues_merge_report.txt"

    parser = argparse.ArgumentParser(
        description="Merge all JSON files from Data_Collector/data/raw/issues into one JSON file.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help=f"Directory containing per-page/per-repo JSON files (default: {default_input})",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=default_output,
        help=f"Output JSON file path (default: {default_output})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on the first malformed JSON/JSONL instead of skipping.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write the merge report file.",
    )
    args = parser.parse_args()

    report_file = None if args.no_report else default_report
    total_files, processed_files, total_records, bad_lines = merge_issues(
        args.input_dir,
        args.output_file,
        strict=args.strict,
        report_file=report_file,
    )
    print(
        "Merged "
        f"{total_records} records from {processed_files}/{total_files} files into: {args.output_file}",
    )
    if bad_lines and not args.strict:
        print(f"Skipped {bad_lines} malformed JSONL lines (see report).", file=sys.stderr)
    if report_file is not None:
        print(f"Report: {report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
