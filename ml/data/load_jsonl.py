from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List


def load_issues_from_jsonl(root: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in Path(root).rglob("*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records
