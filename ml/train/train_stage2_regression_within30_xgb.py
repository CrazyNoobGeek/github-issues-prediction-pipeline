from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List, Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.xgboost
import xgboost as xgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
)
from sklearn.model_selection import TimeSeriesSplit


# ----------------------------
# MLflow bootstrap (ONLY MLflow part)
# ----------------------------
def _configure_mlflow(default_experiment: str) -> str:
    repo = Path(__file__).resolve().parents[2]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    tracking_dir = os.getenv("MLFLOW_TRACKING_DIR", "").strip()

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        if tracking_uri.startswith("file:"):
            p = Path(tracking_uri.replace("file:", "", 1)).expanduser().resolve()
            p.mkdir(parents=True, exist_ok=True)
    else:
        if tracking_dir:
            p = Path(tracking_dir).expanduser()
            if not p.is_absolute():
                p = (repo / p).resolve()
        else:
            p = (repo / "mlruns").resolve()

        p.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"file:{p}")

    exp_name = os.getenv("MLFLOW_EXPERIMENT", default_experiment).strip() or default_experiment
    mlflow.set_experiment(exp_name)
    return exp_name


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

REPO_FEATURES = ["repo_count_train", "repo_median_days_train"]

ALL_FEATURES_ORDER = (
    TABULAR_BASE
    + TTF_FEATURES
    + EXTRA_FEATURES
    + [name for name, _ in LABEL_FLAGS]
    + ["author_is_owner", "author_is_member", "author_is_contributor", "author_is_none", "author_is_other"]
    + REPO_FEATURES
)

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


def time_split(df: pd.DataFrame, train=0.8, dev=0.1, test=0.1):
    if not np.isclose(train + dev + test, 1.0):
        raise ValueError("train+dev+test must sum to 1.0")
    df = df.sort_values("created_ts").reset_index(drop=True)
    n = len(df)
    n_train = int(n * train)
    n_dev = int(n * dev)
    return df.iloc[:n_train].copy(), df.iloc[n_train:n_train + n_dev].copy(), df.iloc[n_train + n_dev:].copy()


def fit_repo_features_train_only(train_df: pd.DataFrame, y_train_days: np.ndarray) -> Dict[str, Any]:
    repo = train_df.get("repo_full_name", pd.Series(["UNKNOWN"] * len(train_df))).astype(str)
    repo_counts = repo.value_counts().to_dict()

    tmp = pd.DataFrame({"repo": repo.values, "y": y_train_days.astype(float)})
    repo_median = tmp.groupby("repo")["y"].median().to_dict()

    global_median = float(np.median(y_train_days))
    global_count = float(np.mean(list(repo_counts.values()))) if repo_counts else 1.0

    return {
        "repo_counts": repo_counts,
        "repo_median": repo_median,
        "global_median": global_median,
        "global_count": global_count,
    }


def apply_repo_features(df: pd.DataFrame, repo_fit: Dict[str, Any]) -> pd.DataFrame:
    repo = df.get("repo_full_name", pd.Series(["UNKNOWN"] * len(df))).astype(str)

    counts = np.array([repo_fit["repo_counts"].get(r, repo_fit["global_count"]) for r in repo], dtype=float)
    med = np.array([repo_fit["repo_median"].get(r, repo_fit["global_median"]) for r in repo], dtype=float)

    return pd.DataFrame({
        "repo_count_train": np.log1p(np.maximum(counts, 0.0)),
        "repo_median_days_train": med,
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
    df = df.copy()

    for c in TABULAR_BASE:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).astype(float)

    if prediction_moment == "t0":
        df["comments"] = 0.0
        df["ttf_hours"] = np.nan

    if "body" in df.columns:
        df["body_missing"] = df["body"].isna().astype(float)
    else:
        df["body_missing"] = 0.0

    ttf = pd.to_numeric(df.get("ttf_hours", np.nan), errors="coerce")
    df["ttf_missing"] = ttf.isna().astype(float)
    ttf_filled = ttf.fillna(ttf_fill_value_train)
    df["ttf_capped_72h"] = np.minimum(ttf_filled, 72.0)
    df["ttf_log1p"] = np.log1p(np.maximum(ttf_filled, 0.0))

    lab = _label_flags(df)
    auth = _author_onehot(df)
    repo_feats = apply_repo_features(df, repo_fit)

    tab = pd.concat([df[TABULAR_BASE + TTF_FEATURES + EXTRA_FEATURES], lab, auth, repo_feats], axis=1)

    for c in ALL_FEATURES_ORDER:
        if c not in tab.columns:
            tab[c] = 0.0
    tab = tab[ALL_FEATURES_ORDER].astype(float)

    bad = set(tab.columns).intersection(FORBIDDEN_INPUT_COLS)
    if bad:
        raise RuntimeError(f"Leakage: forbidden columns found in tabular X: {bad}")

    return tab


def metrics_days(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(y_pred - y_true)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "acc_within_3d": float(np.mean(abs_err <= 3.0)),
        "acc_within_7d": float(np.mean(abs_err <= 7.0)),
        "acc_within_14d": float(np.mean(abs_err <= 14.0)),
    }


def _clamp_preds(pred: np.ndarray, horizon_days: int) -> np.ndarray:
    return np.clip(pred, 0.0, float(horizon_days))


def time_cv_best_iters(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    emb_all: np.ndarray,
    row_to_idx: Dict[str, int],
    params: Dict[str, Any],
    num_boost_round: int,
    early_stopping: int,
    n_splits: int,
    prediction_moment: str,
    use_weights: bool,
) -> List[int]:
    train_df = train_df.sort_values("created_ts").reset_index(drop=True)
    y_train = y_train.astype(float)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_iters: List[int] = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train_df), start=1):
        tr_df = train_df.iloc[tr_idx].copy()
        va_df = train_df.iloc[va_idx].copy()
        y_tr = y_train[tr_idx]
        y_va = y_train[va_idx]

        ttf_tr = pd.to_numeric(tr_df.get("ttf_hours", np.nan), errors="coerce")
        ttf_fill = float(ttf_tr.median(skipna=True)) if ttf_tr.notna().any() else 0.0
        repo_fit = fit_repo_features_train_only(tr_df, y_tr)

        X_tr_tab = build_tabular(tr_df, ttf_fill, repo_fit, prediction_moment).to_numpy()
        X_va_tab = build_tabular(va_df, ttf_fill, repo_fit, prediction_moment).to_numpy()

        X_tr = np.concatenate([X_tr_tab, align_embeddings(tr_df, emb_all, row_to_idx)], axis=1)
        X_va = np.concatenate([X_va_tab, align_embeddings(va_df, emb_all, row_to_idx)], axis=1)

        y_tr_t = np.log1p(y_tr)
        y_va_t = np.log1p(y_va)

        if use_weights:
            w_tr = 1.0 / np.sqrt(1.0 + y_tr)
            w_va = 1.0 / np.sqrt(1.0 + y_va)
        else:
            w_tr = None
            w_va = None

        dtr = xgb.DMatrix(X_tr, label=y_tr_t, weight=w_tr)
        dva = xgb.DMatrix(X_va, label=y_va_t, weight=w_va)

        booster = xgb.train(
            params=params,
            dtrain=dtr,
            num_boost_round=num_boost_round,
            evals=[(dtr, "train"), (dva, "val")],
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )
        best_iters.append(int(getattr(booster, "best_iteration", 0) or 0))

    return best_iters


def main() -> None:
    artifacts_dir = _artifacts_dir()
    df_path = artifacts_dir / "datasets" / "issues_clean_latest.parquet"
    if not df_path.exists():
        raise FileNotFoundError(f"Missing dataset: {df_path}")


    exp_name = _configure_mlflow("github_stage2_within30_reg_xgb_cv")

    train_frac = float(os.getenv("TRAIN_FRAC", "0.8"))
    dev_frac = float(os.getenv("DEV_FRAC", "0.1"))
    test_frac = float(os.getenv("TEST_FRAC", "0.1"))

    horizon_days = int(os.getenv("HORIZON_DAYS", "30"))
    tree_method = os.getenv("TREE_METHOD", "hist")

    n_splits = int(os.getenv("CV_SPLITS", "5"))
    num_boost_round = int(os.getenv("N_ESTIMATORS", "8000"))
    early_stopping = int(os.getenv("EARLY_STOPPING_ROUNDS", "250"))

    prediction_moment = os.getenv("PREDICTION_MOMENT", "latest").lower()
    if prediction_moment not in ("latest", "t0"):
        raise ValueError("PREDICTION_MOMENT must be 'latest' or 't0'")

    use_weights = int(os.getenv("USE_WEIGHTS", "1")) == 1

    params: Dict[str, Any] = {
        "objective": os.getenv("XGB_OBJECTIVE", "reg:pseudohubererror"),
        "eval_metric": "mae",
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

    days = pd.to_numeric(df["days_to_close"], errors="coerce")
    df = df[(days.notna()) & (days <= float(horizon_days))].copy()
    df["days_to_close"] = pd.to_numeric(df["days_to_close"], errors="coerce")
    df = df[df["days_to_close"].notna()].copy()

    if len(df) < 500:
        raise RuntimeError(f"Too few rows for stage2 regression (closed<= {horizon_days}d): n={len(df)}")

    train_df, dev_df, test_df = time_split(df, train=train_frac, dev=dev_frac, test=test_frac)

    emb_all, row_to_idx, embed_model, embed_meta = load_embedding_bundle(artifacts_dir)

    y_train = train_df["days_to_close"].astype(float).to_numpy()
    y_dev = dev_df["days_to_close"].astype(float).to_numpy()
    y_test = test_df["days_to_close"].astype(float).to_numpy()

    best_iters = time_cv_best_iters(
        train_df=train_df,
        y_train=y_train,
        emb_all=emb_all,
        row_to_idx=row_to_idx,
        params=params,
        num_boost_round=num_boost_round,
        early_stopping=early_stopping,
        n_splits=n_splits,
        prediction_moment=prediction_moment,
        use_weights=use_weights,
    )
    best_iter_cv = int(np.median(best_iters))

    ttf_tr = pd.to_numeric(train_df.get("ttf_hours", np.nan), errors="coerce")
    ttf_fill = float(ttf_tr.median(skipna=True)) if ttf_tr.notna().any() else 0.0
    repo_fit = fit_repo_features_train_only(train_df, y_train)

    X_train_tab = build_tabular(train_df, ttf_fill, repo_fit, prediction_moment).to_numpy()
    X_dev_tab = build_tabular(dev_df, ttf_fill, repo_fit, prediction_moment).to_numpy()
    X_test_tab = build_tabular(test_df, ttf_fill, repo_fit, prediction_moment).to_numpy()

    X_train = np.concatenate([X_train_tab, align_embeddings(train_df, emb_all, row_to_idx)], axis=1)
    X_dev = np.concatenate([X_dev_tab, align_embeddings(dev_df, emb_all, row_to_idx)], axis=1)
    X_test = np.concatenate([X_test_tab, align_embeddings(test_df, emb_all, row_to_idx)], axis=1)

    y_train_t = np.log1p(y_train)
    y_dev_t = np.log1p(y_dev)
    y_test_t = np.log1p(y_test)

    if use_weights:
        w_train = 1.0 / np.sqrt(1.0 + y_train)
        w_dev = 1.0 / np.sqrt(1.0 + y_dev)
    else:
        w_train = None
        w_dev = None

    dtrain = xgb.DMatrix(X_train, label=y_train_t, weight=w_train)
    ddev = xgb.DMatrix(X_dev, label=y_dev_t, weight=w_dev)
    dtest = xgb.DMatrix(X_test, label=y_test_t)

    evals_result: Dict[str, Any] = {}

    baseline_med = float(np.median(y_train))
    baseline_test_pred = np.full_like(y_test, baseline_med, dtype=float)
    baseline_test_m = metrics_days(y_test, baseline_test_pred)

    with mlflow.start_run():
        mlflow.set_tag("stage", "stage2")
        mlflow.set_tag("model_family", "xgboost")
        mlflow.set_tag("experiment_name", exp_name)
        mlflow.set_tag("tracking_uri", mlflow.get_tracking_uri())

        mlflow.log_param("task", "stage2_regression_days_to_close_within_horizon")
        mlflow.log_param("horizon_days", horizon_days)
        mlflow.log_param("prediction_moment", prediction_moment)
        mlflow.log_param("cv_splits", n_splits)
        mlflow.log_param("cv_best_iter_median", best_iter_cv)

        mlflow.log_param("embed_model", embed_model)
        mlflow.log_param("embed_shape", embed_meta["shape"])
        mlflow.log_param("ttf_fill_train_median", ttf_fill)
        mlflow.log_param("X_dim", int(X_train.shape[1]))
        mlflow.log_param("tabular_cols", ALL_FEATURES_ORDER)
        mlflow.log_param("num_boost_round", num_boost_round)
        mlflow.log_param("early_stopping_rounds", early_stopping)
        mlflow.log_param("use_weights", int(use_weights))

        for k, v in params.items():
            mlflow.log_param(f"xgb_{k}", v)

        for k, v in baseline_test_m.items():
            mlflow.log_metric(f"baseline_test_{k}", v)

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

        pred_train = _clamp_preds(np.expm1(booster.predict(dtrain)), horizon_days)
        pred_dev = _clamp_preds(np.expm1(booster.predict(ddev)), horizon_days)
        pred_test = _clamp_preds(np.expm1(booster.predict(dtest)), horizon_days)

        train_m = metrics_days(y_train, pred_train)
        dev_m = metrics_days(y_dev, pred_dev)
        test_m = metrics_days(y_test, pred_test)

        for k, v in train_m.items(): mlflow.log_metric(f"train_{k}", v)
        for k, v in dev_m.items(): mlflow.log_metric(f"dev_{k}", v)
        for k, v in test_m.items(): mlflow.log_metric(f"test_{k}", v)

        print("\n=== [STAGE2-30D] REGRESSION METRICS (within horizon) ===")
        print(f"best_iteration: {best_iter} (cv_median={best_iter_cv})")
        print("--- BASELINE (test, train-median) ---")
        for k, v in sorted(baseline_test_m.items()):
            print(f"baseline_test_{k}: {v}")
        print("--- TRAIN ---")
        for k, v in sorted(train_m.items()):
            print(f"train_{k}: {v}")
        print("--- DEV ---")
        for k, v in sorted(dev_m.items()):
            print(f"dev_{k}: {v}")
        print("--- TEST ---")
        for k, v in sorted(test_m.items()):
            print(f"test_{k}: {v}")
        print()

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)

            tr = evals_result.get("train", {}).get("mae", [])
            dv = evals_result.get("dev", {}).get("mae", [])
            if tr and dv:
                plt.figure()
                plt.plot(tr, label="train_mae")
                plt.plot(dv, label="dev_mae")
                plt.legend()
                plt.title("Learning curve (MAE on log1p target)")
                plt.xlabel("round"); plt.ylabel("mae")
                _savefig(p / "learning_curve_mae.png")

            (p / "evals_result.json").write_text(json.dumps(evals_result, indent=2), encoding="utf-8")
            mlflow.log_artifacts(str(p), artifact_path="plots")

        model_dir = _models_dir()
        booster_json = model_dir / f"stage2_within{horizon_days}_booster.json"
        booster.save_model(str(booster_json))

        bundle = {
            "task": "stage2_regression_days_to_close_within_horizon",
            "horizon_days": horizon_days,
            "booster_json": str(booster_json),
            "prediction_moment": prediction_moment,
            "tabular_cols": ALL_FEATURES_ORDER,
            "ttf_fill_train_median": float(ttf_fill),
            "repo_fit": repo_fit,
            "embed_model": embed_model,
            "embed_meta": embed_meta,
            "dataset_parquet": str(df_path),
            "target_transform": "log1p",
            "pred_clamp": [0.0, float(horizon_days)],
            "use_weights": bool(use_weights),
            "version": "stage2_xgb_within30_mlflow_cv_v2",
        }
        out_path = model_dir / f"stage2_within{horizon_days}_xgb.joblib"
        joblib.dump(bundle, out_path)

        mlflow.log_artifact(str(out_path), artifact_path="model_bundle")
        mlflow.log_artifact(str(booster_json), artifact_path="model_bundle")
        try:
            mlflow.xgboost.log_model(booster, name="xgb_model")
        except Exception:
            pass

        print("[ok] stage2 saved:")
        print(" -", out_path)
        print(" -", booster_json)


if __name__ == "__main__":
    main()