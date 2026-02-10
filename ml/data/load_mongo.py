from __future__ import annotations
from typing import Any, Dict, List
from pymongo import MongoClient


def load_issues_from_mongo(uri: str, db: str, collection: str) -> List[Dict[str, Any]]:
    client = MongoClient(uri)
    col = client[db][collection]
    # do not return _id
    return list(col.find({}, {"_id": 0}))
