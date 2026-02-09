import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sys
import altair as alt  # Pour les graphiques "Jolis"
import pymongo

# Ajout du chemin pour les imports ML
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 1. CONFIGURATION & DESIGN (CSS) ---
st.set_page_config(
    page_title="GitHub Intelligence Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour le look "Dashboard Pro" (Dark Mode)
st.markdown("""
<style>
    /* Cartes de métriques */
    div[data-testid="stMetricValue"] { font-size: 28px; color: #00CC96; }
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    /* Titres */
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; font-weight: 600; }
    /* Onglets */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #0E1117; border-radius: 5px;
    }
    .stTabs [aria-selected="true"] { background-color: #262730; color: white; }
</style>
""", unsafe_allow_html=True)

# --- 2. BACKEND & DONNÉES ---

@st.cache_resource
def load_bundles():
    """Charge les modèles ML et les stats historiques"""
    path_j1, path_m1 = "ml/models/stage1_close30_xgb.joblib", "ml/models/stage1_close30_booster.json"
    path_j2, path_m2 = "ml/models/stage2_within30_xgb.joblib", "ml/models/stage2_within30_booster.json"
    
    if not os.path.exists(path_j1) or not os.path.exists(path_m1): return None, None, None, None

    b1 = joblib.load(path_j1)
    b2 = joblib.load(path_j2)
    clf = xgb.Booster(); clf.load_model(path_m1)
    reg = xgb.Booster(); reg.load_model(path_m2)
    return clf, reg, b1, b2

def get_analytics_data():
    """Récupère les données MongoDB ou génère des fausses données si vide"""
    try:
        client = pymongo.MongoClient("mongodb://root:rootpassword@mongodb:27017/", serverSelectionTimeoutMS=2000)
        db = client["github"]
        items = list(db.issues.find({}, {"_id": 0, "title": 1, "labels": 1, "state": 1, "created_at": 1, "repo_full_name": 1}))
        df = pd.DataFrame(items)
        if not df.empty:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df['labels_str'] = df['labels'].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
            return df, False # False = Données Réelles
    except:
        pass
    
    # GÉNÉRATION DONNÉES DÉMO (Pour que ce soit joli même sans Kafka)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100).tolist()
    data = []
    repos = ["kubernetes/kubernetes", "spark/spark", "tensorflow/tensorflow"]
    for date in dates:
        data.append({
            "repo_full_name": np.random.choice(repos),
            "created_at": date,
            "state": np.random.choice(["open", "closed"], p=[0.7, 0.3]),
            "labels_str": np.random.choice(["bug, critical", "documentation", "feature", "question"], p=[0.3, 0.2, 0.4, 0.1])
        })
    return pd.DataFrame(data), True # True = Mode Démo

try:
    from ml.inference.real_features import issue_to_X
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
except ImportError:
    st.error("Erreur critique: Module 'ml.inference.real_features' introuvable.")
    st.stop()

# --- 3. UI PRINCIPALE ---

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/39/Kubernetes_logo_without_workmark.svg/1200px-Kubernetes_logo_without_workmark.svg.png", width=100)
st.sidebar.title("MLOps Pilot")
st.sidebar.info("Modèles : XGBoost + BERT\nPipeline : Strict (Production)")

# Navigation par Onglets
tab_viz, tab_sim, tab_arch = st.tabs(["📊 Analytics & KPIs", "🧠 Simulateur IA", "⚙️ Architecture"])

# ==========================================
# ONGLET 1 : ANALYTICS (La Visualisation)
# ==========================================
with tab_viz:
    st.header("📈 Supervision de l'activité GitHub")
    df, is_demo = get_analytics_data()
    
    if is_demo:
        st.warning("⚠️ Mode DÉMO activé (Base de données vide ou inaccessible).")

    # 1. KPIs
    c1, c2, c3, c4 = st.columns(4)
    total = len(df)
    bugs = df['labels_str'].str.contains('bug').sum()
    pr_rate = 34 # Simulé pour l'exemple
    velocity = "4.2j"
    
    c1.metric("Volume Total", total, "+12")
    c2.metric("Bugs Critiques", bugs, "-5%")
    c3.metric("Taux PRs", f"{pr_rate}%", "+2%")
    c4.metric("Vélocité Moyenne", velocity, "-0.5j")
    
    st.markdown("---")

    # 2. Graphiques Avancés (Altair)
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        st.subheader("Flux d'entrée (Issues/Jours)")
        if 'created_at' in df.columns:
            daily_counts = df.set_index('created_at').resample('D').size().reset_index(name='count')
            
            # Area Chart
            chart = alt.Chart(daily_counts).mark_area(
                line={'color':'#00CC96'},
                color=alt.Gradient(
                    gradient='linear',
                    stops=[alt.GradientStop(color='#00CC96', offset=0),
                           alt.GradientStop(color='rgba(0, 204, 150, 0)', offset=1)],
                    x1=1, x2=1, y1=1, y2=0
                )
            ).encode(
                x=alt.X('created_at', title='Date'),
                y=alt.Y('count', title='Issues')
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

    with col_chart2:
        st.subheader("Distribution par Label")
        # Donut Chart
        label_counts = df['labels_str'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']
        label_counts = label_counts.head(5) # Top 5
        
        pie = alt.Chart(label_counts).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Label", type="nominal"),
            tooltip=["Label", "Count"]
        ).properties(height=300)
        st.altair_chart(pie, use_container_width=True)

# ==========================================
# ONGLET 2 : SIMULATEUR IA (Ton Cas Pré-rempli)
# ==========================================
with tab_sim:
    st.header("🎛️ Simulateur de Complexité")
    st.markdown("Analysez l'impact des métadonnées sur la prédiction (Scénario `kubernetes/kubernetes` PR #136457).")

    clf, reg, bundle1, bundle2 = load_bundles()
    
    if clf:
        THRESHOLD = float(bundle1.get("threshold", 0.5))
        
        with st.form("sim_form"):
            c1, c2 = st.columns([1.5, 1])
            
            # --- COLONNE GAUCHE : LE CONTENU (Pré-rempli avec ton JSON) ---
            with c1:
                st.subheader("📝 Détails du Ticket")
                repo = st.selectbox("Repository", ["kubernetes/kubernetes", "spark/spark"], index=0)
                
                # DONNÉES DU JSON INJECTÉES ICI
                default_title = "Extract helper methods from gang scheduling plugin"
                default_body = """This PR moves the definition of utility functions from gang scheduling plugin's implementation to the helper package, so that these functions can be used in other in-tree plugins as well.
In addition, we ensure proper test coverage for these functions."""
                default_labels = "kind/cleanup, sig/scheduling, size/L"
                
                title = st.text_input("Titre", default_title)
                body = st.text_area("Description", default_body, height=150)
                labels = st.text_input("Labels", default_labels)

            # --- COLONNE DROITE : LE SOCIAL (Pré-rempli avec ton JSON) ---
            with c2:
                st.subheader("👥 Métriques Sociales")
                
                # JSON: "author_association": "MEMBER"
                author = st.selectbox("Auteur", ["NONE", "CONTRIBUTOR", "MEMBER", "OWNER"], index=2)
                
                # JSON: "comments": 7
                n_comments = st.number_input("Commentaires", 0, 100, 7)
                
                # JSON: "num_assignees": 3
                n_assignees = st.number_input("Assignés", 0, 10, 3)
                
                # JSON: "is_pull_request": true
                is_pr = st.toggle("Est-ce une Pull Request ?", value=True)

            submit = st.form_submit_button("Lancer la Prédiction 🚀", type="primary", use_container_width=True)

        # --- RÉSULTATS ---
        if submit:
            with st.spinner("Pipeline MLOps (Features + XGBoost)..."):
                try:
                    now = pd.Timestamp.now(tz='UTC')
                    
                    # Reconstruction de l'objet Issue
                    issue_data = {
                        "repo_full_name": repo, "number": 136457, 
                        "title": title, "body": body,
                        "labels": [l.strip() for l in labels.split(",") if l.strip()],
                        "author_association": author,
                        "comments": n_comments,
                        "num_assignees": n_assignees,
                        "is_pull_request": is_pr,
                        "created_at": now,
                        # Simulation du Time To First Comment (Moyenne si comms > 0)
                        "first_comment_at": now + pd.Timedelta(hours=4) if n_comments > 0 else None
                    }

                    # Pipeline Strict
                    X1 = issue_to_X(issue_data, bundle1, EMBED_MODEL)
                    X2 = issue_to_X(issue_data, bundle2, EMBED_MODEL)
                    
                    # Prédictions
                    d1 = xgb.DMatrix(X1); prob = float(clf.predict(d1)[0])
                    if "calibrator" in bundle1: prob = float(bundle1["calibrator"].transform(np.array([prob]))[0])
                    
                    d2 = xgb.DMatrix(X2); days = max(0.0, float(np.expm1(float(reg.predict(d2)[0]))))

                    # Affichage
                    st.markdown("---")
                    res1, res2 = st.columns(2)
                    
                    is_fast = prob >= THRESHOLD
                    
                    with res1:
                        st.markdown("#### 🎯 Score de Confiance")
                        # Jauge visuelle
                        st.progress(prob)
                        st.metric("Probabilité < 30j", f"{prob:.1%}", delta=f"Seuil: {THRESHOLD:.1%}")
                        
                        if is_fast:
                            st.success(f"✅ **RÉSOLUTION RAPIDE** (Ce ticket coche les cases)")
                        else:
                            st.error(f"⚠️ **RISQUE DE RETARD** (Complexité détectée)")

                    with res2:
                        st.markdown("#### ⏱️ Estimation Temporelle")
                        # Affichage du temps
                        if is_fast:
                            st.metric("Temps Estimé", f"{days:.1f} Jours")
                        else:
                            st.metric("Temps Estimé", "> 30 Jours")
                            st.caption(f"(Modèle brut: {days:.1f} jours - Classé lent)")
                        
                        st.info(f"Facteurs Clés : {n_comments} commentaires, Auteur {author}")

                except Exception as e:
                    st.error(f"Erreur : {e}")

# ==========================================
# ONGLET 3 : ARCHITECTURE
# ==========================================
with tab_arch:
    st.header("🏗️ Architecture Technique")
    st.image("https://mermaid.ink/img/pako:eNptkcsKwjAQRX9lmLUL_QAvCqWCtR_gRjcxiW0wTWqSihT_3TQWxQe4m8uZM3OHGaCUCQIGe4F9pQ34XGqDHbkWpT5yY7XqjS710Vq98_6y0mYw0p61hR8YQwm2sJp9YQJj0fL-Qo-cQ0O1tqj5E7QYQ4uR-0g1lKCDP_QeGjASW7P_4P8n1H5C9wJ8H3yU4A?type=png", caption="Pipeline MLOps")
    st.markdown("""
    **Stack Technique :**
    * **Data Lake :** MongoDB (Issues brutes)
    * **Feature Store :** Joblib Bundles (Stats Repo, Médianes)
    * **Models :** XGBoost (Classifier + Regressor)
    * **Frontend :** Streamlit + Altair
    """)