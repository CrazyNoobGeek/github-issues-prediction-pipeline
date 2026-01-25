from typing import List, Dict, Any
from pymongo import MongoClient


def load_issues_from_mongo(
    uri: str,
    db: str = "github",
    collection: str = "issues_clean"
) -> List[Dict[str, Any]]:
    client = MongoClient(uri)
    col = client[db][collection]
    return list(col.find({}, {"_id": 0}))
