import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Allow running from either repo root (recommended) or from inside `ml/`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.data.load_jsonl import load_issues_from_jsonl
from ml.features.build_features import build_dataframe


def main():
    mlflow.set_experiment("github_issue_resolution_classification")

    candidates = [
        REPO_ROOT / "Data_Collector" / "data" / "raw" / "issues",
        REPO_ROOT / "data" / "raw" / "issues",
        Path.cwd() / "data" / "raw" / "issues",
    ]
    issues_root = next((p for p in candidates if p.exists()), candidates[0])

    records = load_issues_from_jsonl(str(issues_root))
    df = build_dataframe(records)

    X = df[
        ["text", "labels_str", "author_association",
         "comments", "num_assignees", "labels_count",
         "title_len", "body_len", "has_body",
         "created_dow", "created_hour", "ttf_hours"]
    ]
    y = df["resolved_30d"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    pre = ColumnTransformer(
        [
            ("txt", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2), "text"),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), ["labels_str", "author_association"]),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median"))
            ]), ["comments", "num_assignees", "labels_count",
                  "title_len", "body_len", "has_body",
                  "created_dow", "created_hour", "ttf_hours"])
        ]
    )

    clf = LogisticRegression(
        max_iter=5000,
        solver="saga",
        n_jobs=-1
    )


    pipe = Pipeline([("pre", pre), ("clf", clf)])

    with mlflow.start_run():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        mlflow.log_metric("f1", f1_score(y_test, preds))
        mlflow.log_metric("precision", precision_score(y_test, preds))
        mlflow.log_metric("recall", recall_score(y_test, preds))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, proba))

        models_dir = REPO_ROOT / "ml" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, models_dir / "issue_classifier.joblib")
        mlflow.sklearn.log_model(pipe, "model")


if __name__ == "__main__":
    main()