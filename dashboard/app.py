import streamlit as st
import pymongo
import pandas as pd
import time

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="GitHub Issues Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FONCTION DE CONNEXION MONGO (Avec cache pour la performance) ---
@st.cache_resource
def init_connection():
    # On se connecte au service 'mongodb' du réseau Docker
    return pymongo.MongoClient("mongodb://root:rootpassword@mongodb:27017/")

# --- CHARGEMENT DES DONNÉES ---
def get_data():
    client = init_connection()
    db = client["github"]
    items = list(db.issues.find({}, {"_id": 0, "title": 1, "labels": 1, "state": 1, "created_at": 1}))
    return pd.DataFrame(items)

# --- SIDEBAR (Barre latérale) ---
st.sidebar.title("🔮 Navigation")
page = st.sidebar.radio("Aller vers", ["Tableau de Bord", "Prédiction (IA)", "Architecture"])

st.sidebar.markdown("---")
st.sidebar.info("Projet Big Data & MLOps\n\n**M2 IA - Le Mans Université**")

# --- PAGE 1 : TABLEAU DE BORD ---
if page == "Tableau de Bord":
    st.title("📊 Tableau de Bord Analytique")
    st.markdown("Vue en temps réel des données ingérées par le pipeline Kafka/Spark.")

    try:
        # Bouton de rafraîchissement manuel
        if st.button('🔄 Actualiser les données'):
            st.rerun()
        
        df = get_data()
        
        if not df.empty:
            # Métriques Clés (KPIs)
            col1, col2, col3 = st.columns(3)
            
            total_issues = len(df)
            bugs = df['labels'].astype(str).str.contains('bug', case=False).sum()
            features = df['labels'].astype(str).str.contains('enhancement|feature', case=False).sum()
            
            col1.metric("📦 Total Issues Stockées", f"{total_issues:,}")
            col2.metric("🐛 Bugs Identifiés", f"{bugs:,}", delta=f"{round(bugs/total_issues*100, 1)}%")
            col3.metric("✨ Features Requests", f"{features:,}", delta=f"{round(features/total_issues*100, 1)}%")
            
            st.markdown("---")
            
            # Graphiques
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Distribution Bugs vs Features")
                chart_data = pd.DataFrame({
                    'Type': ['Bugs', 'Features', 'Autres'],
                    'Nombre': [bugs, features, total_issues - (bugs + features)]
                })
                st.bar_chart(chart_data.set_index('Type'))
            
            with c2:
                st.subheader("Aperçu des Dernières Données")
                st.dataframe(df.tail(10), use_container_width=True)
                
        else:
            st.warning("La base de données est vide pour le moment. Lancez le pipeline !")

    except Exception as e:
        st.error(f"Erreur de connexion à MongoDB : {e}")
        st.warning("Assurez-vous que le conteneur MongoDB est bien lancé via Docker Compose.")

# --- PAGE 2 : PRÉDICTION (PLACEHOLDER POUR LE MOMENT) ---
elif page == "Prédiction (IA)":
    st.title("🤖 Prédiction Intelligente")
    st.markdown("Ce module utilise un modèle de Machine Learning pour classifier une issue.")
    
    user_input = st.text_area("Entrez le titre ou la description de l'issue :", "Memory leak in pod when restarting")
    
    if st.button("Lancer la prédiction"):
        with st.spinner('Le modèle réfléchit...'):
            time.sleep(1) # Simulation
            # TODO: Ici nous chargerons le vrai modèle .pkl de ton collègue
            
            # Simulation simple pour la démo
            if "bug" in user_input.lower() or "error" in user_input.lower() or "fail" in user_input.lower():
                pred = "🐛 BUG"
                conf = "92%"
                color = "error"
            else:
                pred = "✨ FEATURE / QUESTION"
                conf = "85%"
                color = "success"
                
            st.markdown(f"### Résultat :")
            st.markdown(f":{color}[**{pred}**] (Confiance : {conf})")
            st.info("Note : Ceci est une simulation en attendant l'intégration du modèle entraîné.")

# --- PAGE 3 : ARCHITECTURE ---
elif page == "Architecture":
    st.title("🏗️ Architecture du Pipeline")
    st.markdown("""
    Notre solution repose sur une architecture **Microservices** dockerisée :
    
    1. **Ingestion** : Script Python + API GitHub
    2. **Streaming** : Apache Kafka (Buffer)
    3. **Traitement** : Apache Spark (Structured Streaming)
    4. **Stockage** : MongoDB (NoSQL)
    5. **Visualisation** : Streamlit (Ce dashboard)
    """)
    # Tu pourras ajouter ici l'image de ton schéma d'architecture plus tard