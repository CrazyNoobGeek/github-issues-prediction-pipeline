from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable


def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]], mode: str = "a", encoding: str = "utf-8") -> int:
    """
    Append JSONL by default (mode='a').
    Returns number of written records.
    """
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)

    n = 0
    with open(path, mode, encoding=encoding) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n