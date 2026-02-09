# ml/inference/predict_api_embeddings.py

from __future__ import annotations
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from ml.inference.preprocess_single import issue_to_features

app = FastAPI()

clf_bundle = joblib.load("ml/models/issue_classifier_xgb_emb.joblib")
reg_bundle = joblib.load("ml/models/time_to_close_xgb_emb.joblib")

clf = clf_bundle["xgb_model"]
calibrator = clf_bundle.get("calibrator", None)
reg = reg_bundle["xgb_model"]

EMBED_MODEL = clf_bundle["embed_model"]
THRESHOLD = float(clf_bundle.get("threshold", 0.5))

class Issue(BaseModel):
    repo_full_name: str
    number: int
    title: str = ""
    body: str = ""
    labels: List[str] = []
    author_association: Optional[str] = None
    comments: int = 0
    num_assignees: int = 0
    created_at: str
    updated_at: Optional[str] = None
    closed_at: Optional[str] = None
    first_comment_at: Optional[str] = None
    is_pull_request: bool = False

@app.post("/predict")
def predict(issue: Issue):
    issue_dict: Dict[str, Any] = issue.dict()

    X_tab, X_emb = issue_to_features(issue_dict, embed_model_name=EMBED_MODEL, prefer_gpu=True)
    X = np.concatenate([X_tab, X_emb], axis=1)

    if calibrator is not None:
        prob = float(calibrator.predict_proba(X)[0, 1])
    else:
        prob = float(clf.predict_proba(X)[0, 1])

    days = float(reg.predict(X)[0])

    return {
        "prob_resolved_30d": prob,
        "predicted_label_resolved_30d": int(prob >= THRESHOLD),
        "predicted_days_to_close": days,
        "embed_model": EMBED_MODEL,
        "threshold": THRESHOLD,
        "classifier_version": clf_bundle.get("version"),
        "regressor_version": reg_bundle.get("version"),
    }
