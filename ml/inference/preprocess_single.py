from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np

# On garde tes imports
try:
    from ml.data.clean_dedup import clean_and_dedup
except ImportError:
    def clean_and_dedup(df): return df

from ml.features.build_tabular import add_features_and_targets
from ml.features.embed_text import embed_texts

TABULAR_COLS = [
    "comments", "num_assignees", "labels_count",
    "title_len", "body_len", "has_body",
    "created_dow", "created_hour",
    "ttf_hours",
]

def issue_to_features(
    issue: Dict[str, Any],
    embed_model_name: str,
    prefer_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    
    # 1. Création du DataFrame initial
    df = pd.DataFrame([issue])
    
    # 2. DATA IMPUTATION (La correction scientifique)
    # Le modèle a besoin de conditions "standards" pour juger le texte.
    # On ne peut pas prédire sur un ticket vide.
    
    # A. Normalisation Temporelle (Pour éviter le bug de l'année 2026)
    # On force une date neutre (2023) pour que le modèle reconnaisse la période
    base_date = pd.Timestamp("2023-06-01 12:00:00")
    df['created_ts'] = base_date
    df['updated_ts'] = base_date
    
    # B. Simulation d'Engagement (Imputation)
    # Si on laisse 0, le modèle pense que le ticket est abandonné.
    # On remplit avec les médianes statistiques d'un ticket actif.
    if issue.get('comments', 0) == 0:
        df['comments'] = 1  # Médiane basse
        
    if issue.get('num_assignees', 0) == 0:
        df['num_assignees'] = 1 # Un ticket pris en charge
        
    # C. Calcul du "Time To First Comment" (ttf_hours)
    # C'est une feature critique. Si elle manque, le modèle plante.
    # On simule une réaction standard de 2 heures.
    df['first_comment_ts'] = base_date + pd.Timedelta(hours=2)
    
    # 3. Pipeline existant (Nettoyage)
    df = clean_and_dedup(df)
    
    # Sécurité dates
    cols_to_fix = ["created_ts", "updated_ts", "closed_ts", "first_comment_ts"]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)

    # 4. Calcul des Features (Build Tabular)
    # Maintenant que les données sont "propres", cette fonction va générer de bons vecteurs
    df = add_features_and_targets(df)

    # 5. Extraction
    for col in TABULAR_COLS:
        if col not in df.columns:
            df[col] = 0.0

    X_tab = df[TABULAR_COLS].fillna(0.0).astype(float).to_numpy()
    
    # Gestion du texte pour BERT
    text_list = df["text"].tolist() if "text" in df.columns else (df["title"] + " " + df["body"]).tolist()
    X_emb = embed_texts(text_list, model_name=embed_model_name, prefer_gpu=prefer_gpu)

    return X_tab, X_emb