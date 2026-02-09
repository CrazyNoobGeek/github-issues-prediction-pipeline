from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from ml.features.embed_text import embed_texts

# --- CONFIGURATION STRICTE (Identique à l'entraînement) ---
LABEL_FLAGS = [
    ("has_bug", ["bug"]),
    ("has_docs", ["doc", "documentation"]),
    ("has_feature", ["feature", "enhancement"]),
    ("has_question", ["question", "help wanted"]),
    ("has_security", ["security", "vulnerability", "cve"]),
    ("has_good_first_issue", ["good first issue"]),
    ("has_help_wanted", ["help wanted"]),
]

# Colonnes interdites (Data Leakage)
FORBIDDEN_INPUT_COLS = {
    "days_to_close", "resolved_30d", "is_closed", "closed_at", "closed_ts"
}

def _to_ts(s: Any) -> pd.Timestamp:
    if s is None or s == "":
        return pd.NaT
    return pd.to_datetime(s, utc=True, errors="coerce")

def _author_onehot(author_association: Any) -> Dict[str, float]:
    a = str(author_association or "OTHER").upper()
    return {
        "author_is_owner": float(a == "OWNER"),
        "author_is_member": float(a == "MEMBER"),
        "author_is_contributor": float(a == "CONTRIBUTOR"),
        "author_is_none": float(a == "NONE"),
        "author_is_other": float(a not in {"OWNER", "MEMBER", "CONTRIBUTOR", "NONE"}),
    }

def _label_flags(labels: List[str]) -> Dict[str, float]:
    s = " ".join([str(x) for x in (labels or [])]).lower()
    out: Dict[str, float] = {}
    for name, keywords in LABEL_FLAGS:
        m = False
        for kw in keywords:
            if kw.lower() in s:
                m = True
                break
        out[name] = float(m)
    return out

def _apply_repo_features(issue_repo: str, repo_fit: Dict[str, Any], need: List[str]) -> Dict[str, float]:
    # C'EST ICI QUE LA MAGIE OPÈRE : On utilise les stats historiques du repo
    repo = str(issue_repo or "UNKNOWN")
    repo_counts = repo_fit.get("repo_counts", {})
    global_count = float(repo_fit.get("global_count", 1.0))
    cnt = float(repo_counts.get(repo, global_count))
    
    out: Dict[str, float] = {"repo_count_train": float(np.log1p(max(cnt, 0.0)))}

    if "repo_posrate_train" in need:
        repo_posrate = repo_fit.get("repo_posrate", {})
        global_posrate = float(repo_fit.get("global_posrate", 0.0))
        out["repo_posrate_train"] = float(repo_posrate.get(repo, global_posrate))

    if "repo_median_days_train" in need:
        repo_median = repo_fit.get("repo_median", {})
        global_median = float(repo_fit.get("global_median", 0.0))
        out["repo_median_days_train"] = float(repo_median.get(repo, global_median))

    return out

def _build_tabular_row(issue: Dict[str, Any], bundle: Dict[str, Any]) -> np.ndarray:
    tab_cols: List[str] = list(bundle["tabular_cols"])
    
    # Récupération des métadonnées d'entraînement (CRUCIAL)
    repo_fit = bundle.get("repo_fit", {})
    ttf_fill = float(bundle.get("ttf_fill_train_median", 0.0))

    # Extraction des champs de base
    title = str(issue.get("title") or "")
    body = str(issue.get("body") or "")
    labels = issue.get("labels") or []
    
    created_ts = _to_ts(issue.get("created_at"))
    if pd.isna(created_ts):
        # Fallback pour le live : maintenant
        created_ts = pd.Timestamp.now(tz='UTC')

    # Gestion du moment de prédiction (t0 = création vs latest)
    prediction_moment = str(bundle.get("prediction_moment", "latest")).lower()
    
    comments = float(issue.get("comments") or 0)
    num_assignees = float(issue.get("num_assignees") or 0)
    
    # Calculs temporels rigoureux
    first_comment_ts = _to_ts(issue.get("first_comment_at"))
    ttf_hours = np.nan
    
    if not pd.isna(first_comment_ts):
        ttf_hours = (first_comment_ts - created_ts).total_seconds() / 3600.0
        if ttf_hours < 0: ttf_hours = np.nan

    # Si on prédit à la création (t0), on force à 0 (logique stricte)
    if prediction_moment == "t0":
        comments = 0.0
        ttf_hours = np.nan # Sera remplacé par la médiane d'entraînement

    # Features calculées
    base = {
        "comments": comments,
        "num_assignees": num_assignees,
        "labels_count": float(len(labels)),
        "title_len": float(len(title)),
        "body_len": float(len(body)),
        "has_body": float(1.0 if len(body) > 0 else 0.0),
        "created_dow": float(int(created_ts.weekday())),
        "created_hour": float(int(created_ts.hour)),
        "body_missing": float(1.0 if (issue.get("body") is None) else 0.0),
        
        # TTF Engineering (C'est là que ton ancien code échouait)
        "ttf_missing": float(1.0 if not np.isfinite(ttf_hours) else 0.0),
        "ttf_capped_72h": float(min(ttf_hours if np.isfinite(ttf_hours) else ttf_fill, 72.0)),
        "ttf_log1p": float(np.log1p(max(ttf_hours if np.isfinite(ttf_hours) else ttf_fill, 0.0))),
    }

    # Ajout des flags one-hot
    base.update(_label_flags(labels))
    base.update(_author_onehot(issue.get("author_association")))

    # Ajout des stats par Repo (Indispensable pour la performance)
    need_repo = [c for c in tab_cols if c.startswith("repo_")]
    base.update(_apply_repo_features(issue.get("repo_full_name", "UNKNOWN"), repo_fit, need_repo))

    # Construction du vecteur final dans le BON ordre
    x = np.array([float(base.get(c, 0.0)) for c in tab_cols], dtype=np.float32)
    return x

def issue_to_X(issue: Dict[str, Any], bundle: Dict[str, Any], embed_model_name: str) -> np.ndarray:
    # 1. Tabular features (Méta-données)
    x_tab = _build_tabular_row(issue, bundle).reshape(1, -1)

    # 2. Text Embeddings (Sémantique)
    title = str(issue.get("title") or "")
    body = str(issue.get("body") or "")
    text = (title + " " + body).strip()
    
    # Appel au modèle BERT
    x_emb = embed_texts([text], model_name=embed_model_name, prefer_gpu=False, normalize=True)

    # 3. Concaténation
    X = np.concatenate([x_tab, x_emb.astype(np.float32)], axis=1)
    return X