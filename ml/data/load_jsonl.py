import json
from pathlib import Path
from typing import List, Dict, Any


def load_issues_from_jsonl(root: str) -> List[Dict[str, Any]]:
    records = []
    for path in Path(root).rglob("*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records
