import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sys
import altair as alt
import pymongo
import time
from datetime import datetime, timedelta  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 1. CONFIGURATION & DESIGN (CSS) ---
st.set_page_config(
    page_title="GitHub Smart Triage", # ✅ MODIF : Nom plus pro
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'sim_defaults' not in st.session_state:
    st.session_state['sim_defaults'] = None

# CSS pour le look "Dashboard Pro" (Dark Mode)
st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 28px; color: #00CC96; }
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #0E1117; border-radius: 5px;
    }
    .stTabs [aria-selected="true"] { background-color: #262730; color: white; }
    
    .info-box {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #00CC96;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. BACKEND & DONNÉES ---

@st.cache_resource
def load_bundles():
    """Charge les modèles ML et les stats historiques"""
    path_j1, path_m1 = "../ml/models/stage1_close30_xgb.joblib", "../ml/models/stage1_close30_booster.json"
    path_j2, path_m2 = "../ml/models/stage2_within30_xgb.joblib", "../ml/models/stage2_within30_booster.json"
    
    if not os.path.exists(path_j1):
        path_j1, path_m1 = "ml/models/stage1_close30_xgb.joblib", "ml/models/stage1_close30_booster.json"
        path_j2, path_m2 = "ml/models/stage2_within30_xgb.joblib", "ml/models/stage2_within30_booster.json"

    if not os.path.exists(path_j1) or not os.path.exists(path_m1):
        return None, None, None, None

    b1 = joblib.load(path_j1)
    b2 = joblib.load(path_j2)
    clf = xgb.Booster(); clf.load_model(path_m1)
    reg = xgb.Booster(); reg.load_model(path_m2)
    return clf, reg, b1, b2

def get_analytics_data():
    """Récupère les données MongoDB"""
    try:
        client = pymongo.MongoClient("mongodb://root:rootpassword@mongodb:27017/", serverSelectionTimeoutMS=2000)
        db = client["github"]
        
        items = list(db.issues.find({}, {"_id": 0}))
        
        df = pd.DataFrame(items)
        if not df.empty:
            if 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            
            if 'closed_at' in df.columns:
                df['closed_at'] = pd.to_datetime(df['closed_at'], errors='coerce')

            if 'labels' in df.columns:
                df['labels_str'] = df['labels'].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
            else:
                df['labels_str'] = ""
                
            return df, False 
    except Exception as e:
        pass
    
    # Mode Démo
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100).tolist()
    data = []
    repos = ["kubernetes/kubernetes", "spark/spark", "tensorflow/tensorflow"]
    for i, date in enumerate(dates):
        data.append({
            "number": 1000 + i,
            "title": f"Demo Issue {i}",
            "repo_full_name": np.random.choice(repos),
            "created_at": date,
            "closed_at": date + timedelta(days=np.random.randint(1, 10)) if i % 2 == 0 else None, # Simule des fermetures
            "state": np.random.choice(["open", "closed"], p=[0.7, 0.3]),
            "labels_str": np.random.choice(["bug, critical", "documentation", "feature", "question"], p=[0.3, 0.2, 0.4, 0.1]),
            "body": "Description simulée",
            "user": {"login": "demo_user"},
            "comments": 2,
            "assignees": []
        })
    return pd.DataFrame(data), True

try:
    from ml.inference.real_features import issue_to_X
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
except ImportError:
    st.error("⚠️ Module 'ml.inference.real_features' introuvable. Mode dégradé.")
    def issue_to_X(issue, bundle, model): return np.zeros((1, 100))
    EMBED_MODEL = None

# --- 3. FONCTIONS UTILITAIRES ---

def predict_dataframe(df, clf, reg, bundle1, bundle2):
    """Applique les modèles IA sur toutes les lignes du DataFrame (Sécurisé)"""
    results = []
    
    for index, row in df.iterrows():
        try:
            issue_data = row.to_dict()
            if 'body' not in issue_data or pd.isna(issue_data['body']): issue_data['body'] = ""
            if 'user_login' not in issue_data: issue_data['user_login'] = "unknown"
            
            X1 = issue_to_X(issue_data, bundle1, EMBED_MODEL) 
            d1 = xgb.DMatrix(X1)
            prob = float(clf.predict(d1)[0])
            
            if bundle1 and "calibrator" in bundle1:
                prob = float(bundle1["calibrator"].transform(np.array([prob]))[0])
            
            results.append(prob)
        except Exception:
            results.append(None)
            
    return results

# --- 4. UI PRINCIPALE ---

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/39/Kubernetes_logo_without_workmark.svg/1200px-Kubernetes_logo_without_workmark.svg.png", width=100)
st.sidebar.title("GitHub Smart Triage")
st.sidebar.info("Status: ONLINE 🟢\nFlux: Kafka Live")

if st.sidebar.button("🔄 Rafraîchir"):
    st.cache_resource.clear()
    st.rerun()

tab_viz, tab_sim, tab_arch = st.tabs(["📊 Analytics & KPIs", "🧠 Simulateur IA", "⚙️ Architecture"])

# ==========================================
# ONGLET 1 : ANALYTICS
# ==========================================
with tab_viz:
    st.header("📈 Supervision de l'activité GitHub")
    
    st.markdown("""
    <div class="info-box">
    <b>Bienvenue sur le Dashboard Smart Triage.</b><br>
    Ce panneau analyse en temps réel les tickets (Issues & PRs) entrants. 
    Les indicateurs ci-dessous sont calculés dynamiquement sur la base de données.
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner('Connexion au Data Lake en cours...'):
        df, is_demo = get_analytics_data()
        time.sleep(0.3)
    
    if is_demo:
        st.warning("⚠️ Mode DÉMO activé (Base de données vide ou inaccessible).")

    # --- CALCULS DYNAMIQUES DES KPIs ---
    
    # 1. Volume Total
    total_volume = len(df)
    
    # 2. Delta (Combien reçus aujourd'hui ?)
    now = pd.Timestamp.now()
    if 'created_at' in df.columns:
        # On gère le fuseau horaire si nécessaire, sinon on prend la date naïve
        df_today = df[df['created_at'].dt.date == now.date()]
        count_today = len(df_today)
        delta_str = f"+{count_today}" if count_today > 0 else "0"
    else:
        delta_str = "0"

    # 3. Bugs Critiques (Basé sur le mot 'bug' dans les labels)
    bugs_count = df['labels_str'].str.contains('bug', case=False, na=False).sum()
    bugs_percent = (bugs_count / total_volume * 100) if total_volume > 0 else 0
    
    # 4. Vélocité (Temps moyen Closed - Created)
    velocity_str = "N/A"
    if 'closed_at' in df.columns and 'created_at' in df.columns:
        closed_issues = df[df['closed_at'].notna()]
        if not closed_issues.empty:
            avg_duration = (closed_issues['closed_at'] - closed_issues['created_at']).dt.total_seconds().mean()
            # Conversion en jours
            days = avg_duration / (3600 * 24)
            velocity_str = f"{days:.1f}j"
        else:
            velocity_str = "-" # Pas encore de tickets fermés
    # 5. RATIO OUVERTURE (Calcul Dynamique)
    open_count = df[df['state'] == 'open'].shape[0] if 'state' in df.columns else 0
    open_rate = (open_count / total_volume * 100) if total_volume > 0 else 0

    # --- AFFICHAGE DES KPIs ---
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("Volume Total", total_volume, delta_str, help="Nombre total de tickets dans la base. Le petit chiffre indique les nouveaux tickets du jour.")
    c2.metric("Bugs Identifiés", bugs_count, f"{bugs_percent:.0f}%", help="Nombre de tickets étiquetés comme 'bug'. Le pourcentage représente leur part sur le total.")
    c3.metric("Ratio Ouverture", f"{open_rate:.0f}%", help="% de tickets encore ouverts.")
    c4.metric("Vélocité Moyenne", velocity_str, help="Temps moyen réel écoulé entre l'ouverture et la fermeture d'un ticket.")
    
    st.markdown("---")

    # Graphiques
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        st.subheader("Flux d'entrée (Issues/Jours)")
        if 'created_at' in df.columns:
            daily_counts = df.set_index('created_at').resample('D').size().reset_index(name='count')
            chart = alt.Chart(daily_counts).mark_area(
                line={'color':'#00CC96'},
                color=alt.Gradient(
                    gradient='linear',
                    stops=[alt.GradientStop(color='#00CC96', offset=0),
                           alt.GradientStop(color='rgba(0, 204, 150, 0)', offset=1)],
                    x1=1, x2=1, y1=1, y2=0
                )
            ).encode(x='created_at', y='count').properties(height=300)
            st.altair_chart(chart, use_container_width=True)

    with col_chart2:
        st.subheader("Distribution par Label")
        if 'labels_str' in df.columns:
            label_counts = df['labels_str'].value_counts().reset_index().head(5)
            label_counts.columns = ['Label', 'Count']
            pie = alt.Chart(label_counts).mark_arc(innerRadius=50).encode(
                theta="Count", color="Label", tooltip=["Label", "Count"]
            ).properties(height=300)
            st.altair_chart(pie, use_container_width=True)

    # LIVE MONITORING
    st.markdown("---")
    st.subheader("🔮 Prédictions en Temps Réel")
    st.caption("ℹ️ Cliquez sur une ligne du tableau pour charger l'issue dans le simulateur.")

    if 'created_at' in df.columns:
        df = df.sort_values(by='created_at', ascending=False)
    
    if 'state' in df.columns:
        df_open = df[df['state'] == 'open'].head(20).reset_index(drop=True)
    else:
        df_open = df.head(20).reset_index(drop=True)

    if not df_open.empty:
        clf, reg, b1, b2 = load_bundles()
        
        if clf:
            with st.spinner("L'IA analyse les nouveaux tickets..."):
                df_open['complexite_score'] = predict_dataframe(df_open, clf, reg, b1, b2)
            
            def highlight_risk(val):
                if val is None or pd.isna(val): return ''
                color = '#FF4B4B' if val < 0.26 else '#00CC96' 
                return f'background-color: {color}; color: white'

            # Tableau interactif avec drill-down
            event = st.dataframe(
                df_open[['number', 'title', 'complexite_score', 'created_at']]
                .style.applymap(highlight_risk, subset=['complexite_score'])
                .format({'complexite_score': lambda x: "{:.1%}".format(x) if x is not None else "N/A"}),
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            # Gestion du Clic
            if len(event.selection.rows) > 0:
                selected_index = event.selection.rows[0]
                row = df_open.iloc[selected_index]
                
                st.session_state['sim_defaults'] = {
                    "repo": row.get('repo_full_name', 'kubernetes/kubernetes'),
                    "title": row.get('title', ''),
                    "body": row.get('body', ''),
                    "labels": row.get('labels_str', ''),
                    "author": "MEMBER", 
                    "comments": int(row.get('comments', 0)),
                    "assignees": len(row.get('assignees', [])) if isinstance(row.get('assignees'), list) else 0
                }
                st.toast(f"✅ Issue #{row['number']} chargée ! Allez dans l'onglet Simulateur.", icon="🚀")

            st.caption("🔴 Rouge = Risque de Retard (>30j) | 🟢 Vert = Résolution Rapide")
        else:
            st.warning("Modèles introuvables. Vérifiez le dossier ml/models.")

# ==========================================
# ONGLET 2 : SIMULATEUR IA
# ==========================================
with tab_sim:
    st.header("🎛️ Simulateur de Complexité")
    clf, reg, bundle1, bundle2 = load_bundles()
    
    # Récupération des données cliquées
    defaults = st.session_state.get('sim_defaults')
    
    if not defaults:
        defaults = {
            "repo": "kubernetes/kubernetes",
            "title": "Extract helper methods from gang scheduling plugin",
            "body": "This PR moves utility functions...",
            "labels": "kind/cleanup, sig/scheduling",
            "author": "MEMBER",
            "comments": 7,
            "assignees": 3
        }
    else:
        st.success(f"📝 Données chargées depuis le tableau : **{defaults['title']}**")

    if clf:
        with st.form("sim_form"):
            c1, c2 = st.columns([1.5, 1])
            with c1:
                repo_opts = ["kubernetes/kubernetes", "spark/spark"]
                repo_idx = 0 if "kubernetes" in defaults["repo"] else 1
                repo = st.selectbox("Repository", repo_opts, index=repo_idx)
                
                title = st.text_input("Titre", value=defaults["title"])
                body = st.text_area("Description", value=defaults["body"], height=150)
                labels = st.text_input("Labels", value=defaults["labels"])
            with c2:
                auth_opts = ["NONE", "CONTRIBUTOR", "MEMBER", "OWNER"]
                try: auth_idx = auth_opts.index(defaults["author"])
                except: auth_idx = 2
                
                author = st.selectbox("Auteur", auth_opts, index=auth_idx)
                n_comments = st.number_input("Commentaires", 0, 100, int(defaults["comments"]))
                n_assignees = st.number_input("Assignés", 0, 10, int(defaults["assignees"]))
                is_pr = st.toggle("Pull Request ?", value=True)

            submit = st.form_submit_button("Lancer la Prédiction 🚀", type="primary")

        if submit:
            try:
                # Préparation des données
                issue_data = {
                    "repo_full_name": repo, "number": 123, "title": title, "body": body,
                    "labels": [l.strip() for l in labels.split(",")], "author_association": author,
                    "comments": n_comments, "num_assignees": n_assignees, "is_pull_request": is_pr,
                    "created_at": pd.Timestamp.now(tz='UTC'),
                    "first_comment_at": pd.Timestamp.now(tz='UTC') if n_comments > 0 else None
                }
                
                # Inférence : CLASSIFICATION
                X1 = issue_to_X(issue_data, bundle1, EMBED_MODEL)
                d1 = xgb.DMatrix(X1); prob = float(clf.predict(d1)[0])
                if "calibrator" in bundle1: prob = float(bundle1["calibrator"].transform(np.array([prob]))[0])
                
                # Inférence : REGRESSION (Durée en jours)
                X2 = issue_to_X(issue_data, bundle2, EMBED_MODEL)
                d2 = xgb.DMatrix(X2); days = max(0.0, float(np.expm1(float(reg.predict(d2)[0]))))
                
                # --- BOOST ARTIFICIEL CONSERVÉ COMME DEMANDÉ ---
                boost = max(0.0, 60 - (prob * 100))
                days = days + boost
                # -----------------------------------------------
                st.markdown("---")
                col_res1, col_res2 = st.columns(2)
                
                # Seuil de décision
                THRESHOLD = 0.31
                is_fast = prob >= THRESHOLD

                with col_res1:
                    st.markdown("#### 🎯 Score de Confiance")
                    st.progress(prob)
                    st.metric("Probabilité Quick Fix (<30j)", f"{prob:.1%}", delta=f"Seuil: {THRESHOLD:.1%}")
                    
                    if is_fast:
                        st.success(f"✅ **RÉSOLUTION RAPIDE**\n\nL'IA estime que ce ticket sera traité efficacement.")
                    else:
                        st.error(f"⚠️ **RISQUE DE RETARD**\n\nComplexité détectée. Risque de dépasser 30 jours.")

                with col_res2:
                    st.markdown("#### ⏱️ Estimation Temporelle")
                    if is_fast:
                        st.metric("Temps estimé", f"{days:.1f} Jours")
                    else:
                        st.metric("Temps estimé", "> 30 Jours")
                        st.caption(f"(Modèle brut: {days:.1f} jours - Classé lent)")
                    
                    st.info(f"**Facteurs Clés :** {n_comments} commentaires, Auteur {author}")

            except Exception as e:
                st.error(f"Erreur: {e}")

# ==========================================
# ONGLET 3 : ARCHITECTURE
# ==========================================
with tab_arch:
    st.header("🏗️ Architecture Technique")
    st.markdown("Pipeline: Kafka -> Spark -> MongoDB -> Streamlit")