# ml/train/train_stage1_classifier_30d_xgb.py
from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.xgboost
import xgboost as xgb

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    brier_score_loss,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit


# ----------------------------
# Features (tabular must match Stage2)
# ----------------------------
TABULAR_BASE = [
    "comments", "num_assignees", "labels_count",
    "title_len", "body_len", "has_body",
    "created_dow", "created_hour",
]
TTF_FEATURES = ["ttf_missing", "ttf_capped_72h", "ttf_log1p"]
EXTRA_FEATURES = ["body_missing"]

LABEL_FLAGS = [
    ("has_bug", ["bug"]),
    ("has_docs", ["doc", "documentation"]),
    ("has_feature", ["feature", "enhancement"]),
    ("has_question", ["question", "help wanted"]),
    ("has_security", ["security", "vulnerability", "cve"]),
    ("has_good_first_issue", ["good first issue"]),
    ("has_help_wanted", ["help wanted"]),
]

# Train-only repo features (must be computed using TRAIN only!)
REPO_FEATURES = ["repo_count_train", "repo_posrate_train"]

ALL_FEATURES_ORDER = (
    TABULAR_BASE
    + TTF_FEATURES
    + EXTRA_FEATURES
    + [name for name, _ in LABEL_FLAGS]
    + ["author_is_owner", "author_is_member", "author_is_contributor", "author_is_none", "author_is_other"]
    + REPO_FEATURES
)

# Columns that must NEVER enter X (targets / target-derived / future signals)
FORBIDDEN_INPUT_COLS = {
    "days_to_close", "resolved_30d", "is_closed", "close_within_h", "close_within_H",
    "closed_at", "closed_ts", "updated_at", "updated_ts", "first_comment_at", "first_comment_ts",
    "dedup_ts",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifacts_dir() -> Path:
    return Path(os.getenv("ARTIFACTS_DIR", str(_repo_root() / "ml" / "artifacts"))).resolve()


def _models_dir() -> Path:
    p = _repo_root() / "ml" / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# ----------------------------
# Embeddings (supports pointer embed_meta.json)
# ----------------------------
def load_embedding_bundle(artifacts_dir: Path) -> Tuple[np.ndarray, Dict[str, int], str, Dict[str, Any]]:
    emb_dir = artifacts_dir / "embeddings"
    ptr_path = emb_dir / "embed_meta.json"
    if not ptr_path.exists():
        raise FileNotFoundError(f"Missing {ptr_path}. Run: python -m ml.features.generate_embeddings")

    ptr = json.loads(ptr_path.read_text(encoding="utf-8"))
    if isinstance(ptr, dict) and "latest_meta" in ptr:
        meta_path = Path(ptr["latest_meta"])
        if not meta_path.is_absolute():
            meta_path = (artifacts_dir / meta_path).resolve()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = ptr

    npy_path = Path(meta["npy_path"])
    idx_path = Path(meta["index_path"])
    if not npy_path.is_absolute():
        npy_path = (artifacts_dir / npy_path).resolve()
    if not idx_path.is_absolute():
        idx_path = (artifacts_dir / idx_path).resolve()

    if not npy_path.exists():
        raise FileNotFoundError(f"Embeddings npy not found: {npy_path}")
    if not idx_path.exists():
        raise FileNotFoundError(f"Embeddings index not found: {idx_path}")

    emb = np.load(npy_path)
    row_to_idx: Dict[str, int] = {}
    with idx_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            row_to_idx[str(o["row_id"])] = int(o["idx"])

    embed_model = str(meta.get("model_name", "unknown"))
    embed_meta = {
        "model_name": embed_model,
        "shape": list(emb.shape),
        "dtype": str(emb.dtype),
        "npy_path": str(npy_path),
        "index_path": str(idx_path),
    }
    return emb, row_to_idx, embed_model, embed_meta


def align_embeddings(df: pd.DataFrame, emb_all: np.ndarray, row_to_idx: Dict[str, int]) -> np.ndarray:
    idxs: List[int] = []
    for rid in df["row_id"].astype(str).tolist():
        i = row_to_idx.get(rid)
        if i is None:
            raise ValueError(
                f"Missing embedding for row_id={rid}. "
                f"Embeddings mismatch: re-generate embeddings from the SAME dataset parquet."
            )
        idxs.append(i)
    return emb_all[np.asarray(idxs, dtype=int)]


# ----------------------------
# Split helpers
# ----------------------------
def time_split(df: pd.DataFrame, train=0.8, dev=0.1, test=0.1):
    if not np.isclose(train + dev + test, 1.0):
        raise ValueError("train+dev+test must sum to 1.0")
    df = df.sort_values("created_ts").reset_index(drop=True)
    n = len(df)
    n_train = int(n * train)
    n_dev = int(n * dev)
    return df.iloc[:n_train].copy(), df.iloc[n_train:n_train + n_dev].copy(), df.iloc[n_train + n_dev:].copy()


# ----------------------------
# Train-only repo encoding (NO leakage)
# ----------------------------
def fit_repo_features_train_only(train_df: pd.DataFrame, y_train: np.ndarray) -> Dict[str, Any]:
    repo = train_df.get("repo_full_name", pd.Series(["UNKNOWN"] * len(train_df))).astype(str)
    repo_counts = repo.value_counts().to_dict()

    tmp = pd.DataFrame({"repo": repo.values, "y": y_train.astype(float)})
    repo_posrate = tmp.groupby("repo")["y"].mean().to_dict()

    global_posrate = float(np.mean(y_train))
    global_count = float(np.mean(list(repo_counts.values()))) if repo_counts else 1.0

    return {
        "repo_counts": repo_counts,
        "repo_posrate": repo_posrate,
        "global_posrate": global_posrate,
        "global_count": global_count,
    }


def apply_repo_features(df: pd.DataFrame, repo_fit: Dict[str, Any]) -> pd.DataFrame:
    repo = df.get("repo_full_name", pd.Series(["UNKNOWN"] * len(df))).astype(str)
    counts = np.array([repo_fit["repo_counts"].get(r, repo_fit["global_count"]) for r in repo], dtype=float)
    posrate = np.array([repo_fit["repo_posrate"].get(r, repo_fit["global_posrate"]) for r in repo], dtype=float)

    return pd.DataFrame({
        "repo_count_train": np.log1p(np.maximum(counts, 0.0)),
        "repo_posrate_train": posrate,
    }, index=df.index)


def _author_onehot(df: pd.DataFrame) -> pd.DataFrame:
    a = df.get("author_association", pd.Series(["OTHER"] * len(df))).astype(str).str.upper()
    return pd.DataFrame({
        "author_is_owner": (a == "OWNER").astype(float),
        "author_is_member": (a == "MEMBER").astype(float),
        "author_is_contributor": (a == "CONTRIBUTOR").astype(float),
        "author_is_none": (a == "NONE").astype(float),
        "author_is_other": (~a.isin(["OWNER", "MEMBER", "CONTRIBUTOR", "NONE"])).astype(float),
    }, index=df.index)


def _label_flags(df: pd.DataFrame) -> pd.DataFrame:
    s = df.get("labels_str", pd.Series([""] * len(df))).astype(str).str.lower()
    out = {}
    for name, keywords in LABEL_FLAGS:
        m = np.zeros(len(df), dtype=bool)
        for kw in keywords:
            m |= s.str.contains(kw.lower(), na=False)
        out[name] = m.astype(float)
    return pd.DataFrame(out, index=df.index)


def build_tabular(
    df: pd.DataFrame,
    ttf_fill_value_train: float,
    repo_fit: Dict[str, Any],
    prediction_moment: str,
) -> pd.DataFrame:
    """
    prediction_moment:
      - "latest": use features as present in dataset
      - "t0": creation-time only (removes post-creation signals: comments, ttf)
    """
    df = df.copy()

    # Base numeric features
    for c in TABULAR_BASE:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).astype(float)

    # Strict T0
    if prediction_moment == "t0":
        df["comments"] = 0.0
        df["ttf_hours"] = np.nan

    # Extra
    if "body" in df.columns:
        df["body_missing"] = df["body"].isna().astype(float)
    else:
        df["body_missing"] = 0.0

    # TTF features
    ttf = pd.to_numeric(df.get("ttf_hours", np.nan), errors="coerce")
    df["ttf_missing"] = ttf.isna().astype(float)
    ttf_filled = ttf.fillna(ttf_fill_value_train)
    df["ttf_capped_72h"] = np.minimum(ttf_filled, 72.0)
    df["ttf_log1p"] = np.log1p(np.maximum(ttf_filled, 0.0))

    lab = _label_flags(df)
    auth = _author_onehot(df)
    repo_feats = apply_repo_features(df, repo_fit)

    tab = pd.concat([df[TABULAR_BASE + TTF_FEATURES + EXTRA_FEATURES], lab, auth, repo_feats], axis=1)

    # deterministic ordering
    for c in ALL_FEATURES_ORDER:
        if c not in tab.columns:
            tab[c] = 0.0
    tab = tab[ALL_FEATURES_ORDER].astype(float)

    # forbidden check (extra safety)
    bad = set(tab.columns).intersection(FORBIDDEN_INPUT_COLS)
    if bad:
        raise RuntimeError(f"Leakage: forbidden columns found in tabular X: {bad}")

    return tab


# ----------------------------
# Threshold selection & metrics
# ----------------------------
def best_threshold_by_f1(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y_true, probs)
    f1s = (2 * prec * rec) / (prec + rec + 1e-12)
    i = int(np.nanargmax(f1s))
    t = 0.5 if i >= len(thr) else float(thr[i])
    return t, float(f1s[i])


def cls_metrics(y: np.ndarray, probs: np.ndarray, thr: float) -> Dict[str, float]:
    y = y.astype(int)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    pred = (probs >= thr).astype(int)
    out = {
        "logloss": float(log_loss(y, probs)),
        "brier": float(brier_score_loss(y, probs)),
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, probs))
        out["pr_auc"] = float(average_precision_score(y, probs))
    return out


# ----------------------------
# Time-series CV on TRAIN only (OOF predictions for threshold)
# NOTE: TimeSeriesSplit leaves an initial block never validated -> OOF NaNs are expected.
# We compute threshold on the VALID OOF subset only.
# ----------------------------
def time_cv_oof_predictions(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    emb_all: np.ndarray,
    row_to_idx: Dict[str, int],
    params: Dict[str, Any],
    num_boost_round: int,
    early_stopping: int,
    n_splits: int,
    prediction_moment: str,
) -> Tuple[np.ndarray, List[int], float]:
    """
    Returns:
      - oof_probs (aligned to sorted train_df order; may contain NaNs for initial block)
      - best_iterations per fold
      - oof_coverage = fraction of rows that got a validation prediction
    """
    train_df = train_df.sort_values("created_ts").reset_index(drop=True)
    y_train = y_train.astype(int)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.full(len(train_df), np.nan, dtype=float)
    best_iters: List[int] = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train_df), start=1):
        tr_df = train_df.iloc[tr_idx].copy()
        va_df = train_df.iloc[va_idx].copy()
        y_tr = y_train[tr_idx]
        y_va = y_train[va_idx]

        # Train-only fits
        ttf_tr = pd.to_numeric(tr_df.get("ttf_hours", np.nan), errors="coerce")
        ttf_fill = float(ttf_tr.median(skipna=True)) if ttf_tr.notna().any() else 0.0
        repo_fit = fit_repo_features_train_only(tr_df, y_tr)

        X_tr_tab = build_tabular(tr_df, ttf_fill, repo_fit, prediction_moment).to_numpy()
        X_va_tab = build_tabular(va_df, ttf_fill, repo_fit, prediction_moment).to_numpy()

        X_tr = np.concatenate([X_tr_tab, align_embeddings(tr_df, emb_all, row_to_idx)], axis=1)
        X_va = np.concatenate([X_va_tab, align_embeddings(va_df, emb_all, row_to_idx)], axis=1)

        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va, label=y_va)

        booster = xgb.train(
            params=params,
            dtrain=dtr,
            num_boost_round=num_boost_round,
            evals=[(dtr, "train"), (dva, "val")],
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )
        best_iters.append(int(getattr(booster, "best_iteration", 0) or 0))

        oof[va_idx] = booster.predict(dva)

    coverage = float(np.mean(~np.isnan(oof)))
    return oof, best_iters, coverage


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    artifacts_dir = _artifacts_dir()
    df_path = artifacts_dir / "datasets" / "issues_clean_latest.parquet"
    if not df_path.exists():
        raise FileNotFoundError(f"Missing dataset: {df_path}")

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "github_stage1_close30_xgb_cv"))

    # Outer split
    train_frac = float(os.getenv("TRAIN_FRAC", "0.8"))
    dev_frac = float(os.getenv("DEV_FRAC", "0.1"))
    test_frac = float(os.getenv("TEST_FRAC", "0.1"))

    horizon_days = int(os.getenv("HORIZON_DAYS", "30"))
    tree_method = os.getenv("TREE_METHOD", "hist")

    # CV controls (inner split on TRAIN only)
    n_splits = int(os.getenv("CV_SPLITS", "5"))

    num_boost_round = int(os.getenv("N_ESTIMATORS", "8000"))
    early_stopping = int(os.getenv("EARLY_STOPPING_ROUNDS", "250"))

    calibrate = int(os.getenv("CALIBRATE", "1")) == 1

    prediction_moment = os.getenv("PREDICTION_MOMENT", "latest").lower()  # latest | t0
    if prediction_moment not in ("latest", "t0"):
        raise ValueError("PREDICTION_MOMENT must be 'latest' or 't0'")

    # XGB params (regularized defaults)
    params: Dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(os.getenv("MAX_DEPTH", "4")),
        "eta": float(os.getenv("LEARNING_RATE", "0.03")),
        "subsample": float(os.getenv("SUBSAMPLE", "0.8")),
        "colsample_bytree": float(os.getenv("COLSAMPLE_BYTREE", "0.8")),
        "min_child_weight": float(os.getenv("MIN_CHILD_WEIGHT", "8.0")),
        "lambda": float(os.getenv("REG_LAMBDA", "10.0")),
        "alpha": float(os.getenv("REG_ALPHA", "0.0")),
        "tree_method": tree_method,
        "seed": int(os.getenv("SEED", "42")),
    }
    if int(os.getenv("USE_LOSSGUIDE", "1")) == 1:
        params["grow_policy"] = "lossguide"
        params["max_leaves"] = int(os.getenv("MAX_LEAVES", "64"))

    df = pd.read_parquet(df_path)
    for c in ["row_id", "created_ts", "days_to_close"]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in dataset")

    # Target: close within horizon
    days = pd.to_numeric(df["days_to_close"], errors="coerce")
    df["close_within_h"] = ((days.notna()) & (days <= float(horizon_days))).astype(int)

    train_df, dev_df, test_df = time_split(df, train=train_frac, dev=dev_frac, test=test_frac)

    y_train = train_df["close_within_h"].astype(int).to_numpy()
    y_dev = dev_df["close_within_h"].astype(int).to_numpy()
    y_test = test_df["close_within_h"].astype(int).to_numpy()

    # class imbalance handling (train only)
    pos = max(1, int(np.sum(y_train == 1)))
    neg = max(1, int(np.sum(y_train == 0)))
    params["scale_pos_weight"] = float(os.getenv("SCALE_POS_WEIGHT", str(float(neg / pos))))

    # Embeddings
    emb_all, row_to_idx, embed_model, embed_meta = load_embedding_bundle(artifacts_dir)

    # -------- Inner CV on TRAIN: OOF predictions + threshold on VALID OOF subset --------
    oof_probs, best_iters, oof_cov = time_cv_oof_predictions(
        train_df=train_df,
        y_train=y_train,
        emb_all=emb_all,
        row_to_idx=row_to_idx,
        params=params,
        num_boost_round=num_boost_round,
        early_stopping=early_stopping,
        n_splits=n_splits,
        prediction_moment=prediction_moment,
    )
    valid = ~np.isnan(oof_probs)
    if valid.sum() < 200:
        raise RuntimeError(
            f"Too few OOF predictions to choose a stable threshold. "
            f"valid_oof={int(valid.sum())}, coverage={oof_cov:.3f}. "
            f"Try reducing CV_SPLITS or increasing dataset size."
        )
    thr, oof_best_f1 = best_threshold_by_f1(y_train[valid], oof_probs[valid])
    best_iter_cv = int(np.median(best_iters))

    # -------- Final training: TRAIN -> early stop on DEV --------
    ttf_tr = pd.to_numeric(train_df.get("ttf_hours", np.nan), errors="coerce")
    ttf_fill = float(ttf_tr.median(skipna=True)) if ttf_tr.notna().any() else 0.0
    repo_fit = fit_repo_features_train_only(train_df, y_train)

    X_train_tab = build_tabular(train_df, ttf_fill, repo_fit, prediction_moment).to_numpy()
    X_dev_tab = build_tabular(dev_df, ttf_fill, repo_fit, prediction_moment).to_numpy()
    X_test_tab = build_tabular(test_df, ttf_fill, repo_fit, prediction_moment).to_numpy()

    X_train = np.concatenate([X_train_tab, align_embeddings(train_df, emb_all, row_to_idx)], axis=1)
    X_dev = np.concatenate([X_dev_tab, align_embeddings(dev_df, emb_all, row_to_idx)], axis=1)
    X_test = np.concatenate([X_test_tab, align_embeddings(test_df, emb_all, row_to_idx)], axis=1)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    ddev = xgb.DMatrix(X_dev, label=y_dev)
    dtest = xgb.DMatrix(X_test, label=y_test)

    evals_result: Dict[str, Any] = {}

    with mlflow.start_run():
        mlflow.log_param("task", "stage1_classifier_close_within_horizon")
        mlflow.log_param("horizon_days", horizon_days)
        mlflow.log_param("prediction_moment", prediction_moment)
        mlflow.log_param("cv_splits", n_splits)
        mlflow.log_param("cv_oof_coverage", oof_cov)
        mlflow.log_param("cv_best_iter_median", best_iter_cv)
        mlflow.log_param("threshold_from_oof", float(thr))
        mlflow.log_metric("oof_best_f1_valid", float(oof_best_f1))

        mlflow.log_param("embed_model", embed_model)
        mlflow.log_param("embed_shape", embed_meta["shape"])
        mlflow.log_param("ttf_fill_train_median", ttf_fill)
        mlflow.log_param("X_dim", int(X_train.shape[1]))
        mlflow.log_param("tabular_cols", ALL_FEATURES_ORDER)

        for k, v in params.items():
            mlflow.log_param(f"xgb_{k}", v)

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (ddev, "dev")],
            early_stopping_rounds=early_stopping,
            evals_result=evals_result,
            verbose_eval=False,
        )
        best_iter = int(getattr(booster, "best_iteration", 0) or 0)
        mlflow.log_param("best_iteration", best_iter)

        p_train = booster.predict(dtrain)
        p_dev = booster.predict(ddev)
        p_test = booster.predict(dtest)

        calibrator: Optional[IsotonicRegression] = None
        if calibrate:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(p_dev, y_dev)  # fit on DEV only
            p_train = calibrator.transform(p_train)
            p_dev = calibrator.transform(p_dev)
            p_test = calibrator.transform(p_test)
        mlflow.log_param("calibrate_isotonic", int(calibrate))

        # Use threshold from OOF valid subset
        train_m = cls_metrics(y_train, p_train, thr)
        dev_m = cls_metrics(y_dev, p_dev, thr)
        test_m = cls_metrics(y_test, p_test, thr)

        for k, v in train_m.items(): mlflow.log_metric(f"train_{k}", v)
        for k, v in dev_m.items(): mlflow.log_metric(f"dev_{k}", v)
        for k, v in test_m.items(): mlflow.log_metric(f"test_{k}", v)

        print("\n=== [STAGE1-30D] CLASSIFIER METRICS ===")
        print(f"best_iteration: {best_iter} (cv_median={best_iter_cv}, oof_cov={oof_cov:.3f})")
        print(f"threshold (from OOF-valid): {thr:.4f}")
        print("--- TRAIN ---")
        for k, v in sorted(train_m.items()): print(f"train_{k}: {v}")
        print("--- DEV ---")
        for k, v in sorted(dev_m.items()): print(f"dev_{k}: {v}")
        print("--- TEST ---")
        for k, v in sorted(test_m.items()): print(f"test_{k}: {v}")
        print()

        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp)

            tr = evals_result.get("train", {}).get("logloss", [])
            dv = evals_result.get("dev", {}).get("logloss", [])
            if tr and dv:
                plt.figure()
                plt.plot(tr, label="train_logloss")
                plt.plot(dv, label="dev_logloss")
                plt.legend()
                plt.title("Learning curve (logloss)")
                plt.xlabel("round"); plt.ylabel("logloss")
                _savefig(pdir / "learning_curve_logloss.png")

            def plot_roc_pr(y, prob, split: str):
                if len(np.unique(y)) != 2:
                    return
                fpr, tpr, _ = roc_curve(y, prob)
                plt.figure()
                plt.plot(fpr, tpr)
                plt.xlabel("FPR"); plt.ylabel("TPR")
                plt.title(f"ROC ({split})")
                _savefig(pdir / f"roc_{split}.png")

                prec, rec, _ = precision_recall_curve(y, prob)
                plt.figure()
                plt.plot(rec, prec)
                plt.xlabel("Recall"); plt.ylabel("Precision")
                plt.title(f"PR curve ({split})")
                _savefig(pdir / f"pr_{split}.png")

            plot_roc_pr(y_train, p_train, "train")
            plot_roc_pr(y_dev, p_dev, "dev")
            plot_roc_pr(y_test, p_test, "test")

            for split, y, prob in [("train", y_train, p_train), ("dev", y_dev, p_dev), ("test", y_test, p_test)]:
                pred = (prob >= thr).astype(int)
                cm = confusion_matrix(y, pred)
                plt.figure()
                plt.imshow(cm)
                plt.title(f"Confusion matrix ({split})")
                plt.xlabel("Pred"); plt.ylabel("True")
                for (i, j), v in np.ndenumerate(cm):
                    plt.text(j, i, str(v), ha="center", va="center")
                _savefig(pdir / f"confusion_{split}.png")

            (pdir / "evals_result.json").write_text(json.dumps(evals_result, indent=2), encoding="utf-8")
            mlflow.log_artifacts(str(pdir), artifact_path="plots")

        # Save
        model_dir = _models_dir()
        booster_json = model_dir / f"stage1_close{horizon_days}_booster.json"
        booster.save_model(str(booster_json))

        bundle = {
            "task": "stage1_classifier_close_within_horizon",
            "horizon_days": horizon_days,
            "booster_json": str(booster_json),
            "threshold": float(thr),
            "threshold_source": "oof_valid_f1",
            "oof_coverage": float(oof_cov),
            "calibrate_isotonic": bool(calibrate),
            "calibrator": calibrator,
            "prediction_moment": prediction_moment,
            "tabular_cols": ALL_FEATURES_ORDER,
            "ttf_fill_train_median": float(ttf_fill),
            "repo_fit": repo_fit,
            "embed_model": embed_model,
            "embed_meta": embed_meta,
            "dataset_parquet": str(df_path),
            "version": "stage1_xgb_close30_mlflow_cv_v2",
        }
        out_path = model_dir / f"stage1_close{horizon_days}_xgb.joblib"
        joblib.dump(bundle, out_path)

        mlflow.log_artifact(str(out_path), artifact_path="model_bundle")
        mlflow.log_artifact(str(booster_json), artifact_path="model_bundle")
        try:
            mlflow.xgboost.log_model(booster, name="xgb_model")
        except Exception:
            pass

        print("[ok] stage1 saved:")
        print(" -", out_path)
        print(" -", booster_json)


if __name__ == "__main__":
    main()