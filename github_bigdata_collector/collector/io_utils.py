import json
import os
from typing import Any, Dict, Iterable, Optional


def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]], encoding: str = "utf-8") -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)

    with open(path, "w", encoding=encoding) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

import os
import json
from typing import Any, Dict, List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")