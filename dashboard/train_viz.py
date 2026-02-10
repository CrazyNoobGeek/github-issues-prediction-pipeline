import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Audit Qualité ML - JSONL", page_icon="🔬", layout="wide")

@st.cache_data
def load_and_preprocess():
    # volumes: - ./issues_merged.json:/app/training_data.jsonl
    file_path = "issues_merged.json" 
    
    if not os.path.exists(file_path):
        st.error(f"Fichier '{file_path}' introuvable. Vérifiez le montage du volume Docker.")
        return None

    try:
        # Lecture du format JSON Lines (lines=True)
        # On limite à 100 000 lignes pour garantir la fluidité de la démo
        df = pd.read_json(file_path, lines=True, nrows=100000)
        st.toast(f"✅ 100,000 lignes chargées depuis {file_path}")
    except Exception as e:
        st.error(f"Erreur de lecture JSONL : {e}")
        return None

    # Nettoyage des colonnes (Gestion de la casse et espaces)
    df.columns = [c.strip().lower() for c in df.columns]

    # Conversion des dates (on utilise les colonnes déjà calculées dans JSONL)
    if 'created_at' in df.columns and 'closed_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['closed_at'] = pd.to_datetime(df['closed_at'])
        
        if 'days_to_close' not in df.columns:
            df['days_to_close'] = (df['closed_at'] - df['created_at']).dt.total_seconds() / 86400
        
        df['days_to_close'] = df['days_to_close'].clip(lower=0)
        df['resolved_30d'] = (df['days_to_close'] <= 30).astype(int)
        df['created_month'] = df['created_at'].dt.strftime('%Y-%m')
    
    return df

df = load_and_preprocess()

if df is not None:
    st.title("📊 Audit Scientifique du Dataset (Format JSONL)")
    st.markdown(f"Analyse de l'échantillon d'entraînement (**{len(df):,}** issues).")

    # --- SECTION 1 : KPIs ---
    st.header("1. Performance Globale")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Issues Analysées", f"{len(df):,}")
    c2.metric("MTTR Moyen", f"{df['days_to_close'].mean():.1f} j")
    c3.metric("MTTR Médian", f"{df['days_to_close'].median():.1f} j")
    c4.metric("SLA < 30j", f"{(df['resolved_30d'].mean()*100):.1f}%")

    st.divider()

    # --- SECTION 2 : GRAPHIQUES ML ---
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("⚖️ Déséquilibre de Classes")
        # Comparaison 0 (Lent) vs 1 (Rapide)
        fig_class = px.bar(df['resolved_30d'].value_counts().reset_index(), 
                          x='resolved_30d', y='count', color='resolved_30d',
                          labels={'resolved_30d': 'Statut (1=Rapide, 0=Lent)', 'count': 'Nombre'},
                          color_discrete_map={1: '#00CC96', 0: '#FF4B4B'},
                          template="plotly_dark")
        st.plotly_chart(fig_class, use_container_width=True)

    with col_b:
        st.subheader("🦖 Distribution 'Long Tail'")
        # Histogramme en Log Scale pour voir les tickets qui durent des années
        fig_hist = px.histogram(df, x="days_to_close", nbins=50, log_y=True,
                               title="Temps de fermeture (Échelle Logarithmique)",
                               template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)

    # --- SECTION 3 : EXPLORATEUR ---
    with st.expander("🔍 Voir les données brutes du JSONL"):
        st.dataframe(df.head(100))