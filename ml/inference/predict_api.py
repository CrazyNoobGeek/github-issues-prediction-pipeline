from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

clf = joblib.load(REPO_ROOT / "ml" / "models" / "issue_classifier.joblib")
reg = joblib.load(REPO_ROOT / "ml" / "models" / "time_to_close_regressor.joblib")

app = FastAPI()


class Issue(BaseModel):
    title: str = ""
    body: str = ""
    labels: List[str] = []
    author_association: Optional[str] = None
    comments: int = 0
    num_assignees: int = 0
    created_dow: int = 0
    created_hour: int = 0
    ttf_hours: Optional[float] = None


@app.post("/predict")
def predict(issue: Issue):
    X = pd.DataFrame([{
        "text": issue.title + " " + issue.body,
        "labels_str": " ".join(issue.labels),
        "author_association": issue.author_association,
        "comments": issue.comments,
        "num_assignees": issue.num_assignees,
        "labels_count": len(issue.labels),
        "title_len": len(issue.title),
        "body_len": len(issue.body),
        "has_body": int(len(issue.body) > 0),
        "created_dow": issue.created_dow,
        "created_hour": issue.created_hour,
        "ttf_hours": issue.ttf_hours
    }])

    return {
        "prob_resolved_30d": float(clf.predict_proba(X)[0, 1]),
        "predicted_days_to_close": float(reg.predict(X)[0])
    }