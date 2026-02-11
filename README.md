# 🚀 GitHub Issues Prediction Pipeline

[![Spark](https://img.shields.io/badge/Apache%20Spark-3.5.0-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-3.5-231F20?logo=apachekafka&logoColor=white)](https://kafka.apache.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-47A248?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

> **Pipeline Big Data temps réel** pour la collecte, le traitement et la prédiction du temps de résolution des issues GitHub.

---

## 📋 Table des Matières

- [Architecture](#-architecture)
- [Fonctionnalités](#-fonctionnalités)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du Projet](#-structure-du-projet)
- [Services](#-services)
- [Machine Learning](#-machine-learning)
- [Monitoring](#-monitoring)
- [Troubleshooting](#-troubleshooting)
- [Contributeurs](#-contributeurs)

---

## 🏗 Architecture

```mermaid
graph LR
    A[GitHub API] -->|REST| B(Kafka Producer)
    B -->|Push| C{Apache Kafka}
    C -->|Stream| D[Spark Streaming]
    D -->|Clean & Process| E[(MongoDB)]
    E -->|Read| F[Dashboard Streamlit]
    G[ML Models] -->|Predict| F
    G <-->|Track| H[MLflow]
```

### Flux de Données

1. **Collecte** : Le collecteur Python interroge l'API GitHub pour récupérer les issues.
2. **Ingestion** : Les données sont envoyées vers Kafka (topic: `github.issues.raw`).
3. **Traitement** : Spark Structured Streaming consomme, nettoie et enrichit les données.
4. **Stockage** : Les données traitées sont persistées dans MongoDB (upsert).
5. **Prédiction** : Les modèles ML prédisent le temps de résolution des issues.
6. **Visualisation** : Dashboard Streamlit pour l'analyse et le monitoring.

---

## ✨ Fonctionnalités

- ✅ Collecte temps réel des issues GitHub via API REST
- ✅ Streaming avec Kafka pour l'ingestion haute performance
- ✅ Traitement Spark avec nettoyage et déduplication
- ✅ Stockage MongoDB avec upsert (pas de doublons)
- ✅ Modèles ML pour prédire le temps de résolution
- ✅ MLflow pour le tracking des expériences
- ✅ Dashboard interactif avec Streamlit
- ✅ Architecture Docker entièrement conteneurisée

---

## 📦 Prérequis

- Docker >= 24.0
- Docker Compose >= 2.20
- Git >= 2.40
- GitHub Personal Access Token (pour l'API GitHub)

### Ressources recommandées

| Ressource | Minimum | Recommandé |
|-----------|---------|------------|
| RAM       | 8 GB    | 16 GB      |
| CPU       | 4 cores | 8 cores    |
| Disque    | 20 GB   | 50 GB      |

---

## 🛠 Installation

### 1. Cloner le repository

```bash
git clone https://github.com/your-username/github-issues-prediction-pipeline.git
cd github-issues-prediction-pipeline
```

### 2. Configurer les variables d'environnement

Copiez le fichier d'exemple et configurez vos clés :

```bash
cp .env.example .env
```

Éditez le fichier `.env` :

```ini
# GitHub API
GITHUB_TOKEN=ghp_your_personal_access_token

# MongoDB
MONGO_INITDB_ROOT_USERNAME=root
MONGO_INITDB_ROOT_PASSWORD=rootpassword
MONGODB_URI=mongodb://root:rootpassword@mongodb:27017/

# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5001
```

### 3. Lancer l'infrastructure

```bash
# Démarrer tous les services
docker compose up -d

# Vérifier que tous les services sont running
docker compose ps
```

### 4. Initialiser Kafka topics

```bash
docker exec -it kafka kafka-topics --create \
  --topic github.issues.raw \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1
```

---

## 🚀 Utilisation

### Démarrer le pipeline complet

1. **Lancer l'infrastructure** (si ce n'est pas déjà fait) :

```bash
docker compose up -d
```

2. **Démarrer le job Spark Streaming** :

```bash
./spark/submit.sh
```

3. **Lancer le collecteur GitHub** :

```bash
docker compose up -d kafka-producer
```

4. **Accéder au dashboard** : Ouvrez http://localhost:8501

### Commandes utiles

```bash
# Voir les logs en temps réel
docker compose logs -f spark-master

# Arrêter tous les services
docker compose down

# Reconstruire les images
docker compose up -d --build

# Vérifier l'état de Kafka
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
```

---

## 📁 Structure du Projet

```
github-issues-prediction-pipeline/
│
├── 📄 docker-compose.yml          # Orchestration des services
├── 📄 .env                        # Variables d'environnement
├── 📄 requirements.txt            # Dépendances Python globales
│
├── 📂 Data_Collector/             # Module de collecte GitHub
│   ├── github_collector/          # Client API GitHub
│   │   ├── __init__.py
│   │   ├── github_client.py       # Client HTTP GitHub
│   │   ├── issues.py              # Collecte des issues
│   │   └── state.py               # Gestion de l'état
│   ├── config.json                # Configuration des repos
│   ├── run_collector.py           # Point d'entrée collecteur
│   └── collect_issues.sh          # Script shell de collecte
│
├── 📂 github_bigdata_pipeline/    # Pipeline Kafka
│   └── collector/
│       ├── kafka_producer.py      # Producteur Kafka
│       └── run_collector_kafka.py # Collecteur → Kafka
│
├── 📂 spark/                      # Jobs Spark
│   ├── process_stream.py          # Spark Structured Streaming
│   ├── submit.sh                  # Script de soumission
│   └── requirements.txt           # Dépendances Spark
│
├── 📂 ml/                         # Machine Learning
│   ├── train_model.py             # Entraînement des modèles
│   ├── predict.py                 # Prédictions
│   └── feature_engineering.py     # Création de features
│
├── 📂 dashboard/                  # Interface utilisateur
│   ├── app.py                     # Dashboard principal
│   ├── train_viz.py               # Visualisation données training
│   ├── Dockerfile                 # Image Docker Streamlit
│   └── requirements.txt           # Dépendances dashboard
│
├── 📂 mongo/                      # Configuration MongoDB
│   └── init.js                    # Scripts d'initialisation
│
├── 📂 data/                       # Données locales
│   └── state/                     # État du collecteur
│
├── 📂 mlflow-data/                # Artefacts MLflow
└── 📂 mlflow-db/                  # Base de données MLflow
```

---

## 🐳 Services

| Service      | Port  | Description        | URL                      |
|--------------|-------|--------------------|--------------------------|
| Kafka        | 9092  | Message broker     | `localhost:9092`         |
| Zookeeper    | 2181  | Coordination Kafka | `localhost:2181`         |
| MongoDB      | 27017 | Base de données    | `localhost:27017`        |
| Spark Master | 8080  | Cluster manager    | http://localhost:8080    |
| Spark Worker | 8081  | Worker node        | http://localhost:8081    |
| MLflow       | 5001  | Tracking server    | http://localhost:5001    |
| Dashboard    | 8501  | App principale     | http://localhost:8501    |
| Train Viz    | 8502  | Visualisation training | http://localhost:8502 |
| Kafka UI     | 8082  | Interface Kafka    | http://localhost:8082    |

---

## 🤖 Machine Learning

### Modèles disponibles

| Modèle           | Description                    | Métrique |
|------------------|--------------------------------|----------|
| RandomForest     | Classification temps résolution | F1-Score |
| XGBoost          | Gradient boosting              | RMSE     |
| LinearRegression | Baseline régression            | MAE      |

### Features utilisées

```python
features = [
    'title_length',        # Longueur du titre
    'body_length',         # Longueur du corps
    'labels_count',        # Nombre de labels
    'comments_count',      # Nombre de commentaires
    'has_assignee',        # Présence d'assigné
    'is_bug',              # Label "bug" présent
    'is_enhancement',      # Label "enhancement" présent
    'hour_of_day',         # Heure de création
    'day_of_week',         # Jour de la semaine
]
```

### Entraîner un modèle

**Via Docker :**

```bash
docker exec -it dashboard python -m ml.train_model
```

**Localement :**

```bash
python ml/train_model.py --experiment-name "github-issues-v1"
```

### Tracking avec MLflow

Accédez à http://localhost:5001 pour :

- Comparer les runs
- Visualiser les métriques
- Télécharger les artefacts
- Déployer les modèles

---

## 📊 Monitoring

### Spark UI

- **Master UI** : http://localhost:8080
- **Job UI** : http://localhost:4040 (disponible uniquement pendant l'exécution)

### Kafka

**Lister les topics :**

```bash
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
```

**Voir le Consumer Group Lag :**

```bash
docker exec -it kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --describe --group spark-streaming-group
```

### MongoDB

**Accéder au shell :**

```bash
docker exec -it mongodb mongosh -u root -p rootpassword
```

**Compter les documents :**

```javascript
use github
db.issues.countDocuments()
```

---

## 🔌 API Endpoints (Streamlit)

| Route      | Description                   |
|------------|-------------------------------|
| `/`        | Page principale avec statistiques |
| `/predict` | Interface de prédiction       |
| `/data`    | Exploration des données       |

---

## 🔧 Troubleshooting

<details>
<summary><b>❌ Kafka ne démarre pas</b></summary>

```bash
# Vérifier les logs
docker logs kafka

# Redémarrer Zookeeper puis Kafka
docker compose restart zookeeper
docker compose restart kafka
```

</details>

<details>
<summary><b>❌ Spark job échoue</b></summary>

```bash
# Vérifier les logs du master
docker logs spark-master

# Vérifier la mémoire disponible
docker stats

# Augmenter la mémoire dans submit.sh
# --conf spark.executor.memory=4g
```

</details>

<details>
<summary><b>❌ MongoDB connection refused</b></summary>

```bash
# Vérifier que MongoDB est running
docker compose ps mongodb

# Tester la connexion
docker exec -it mongodb mongosh -u root -p rootpassword --eval "db.stats()"
```

</details>

<details>
<summary><b>❌ Dashboard Streamlit ne charge pas</b></summary>

```bash
# Vérifier les logs
docker logs train-data-audit

# Reconstruire l'image
docker compose up -d --build train-viz
```

</details>

---

## 📈 Roadmap

- [ ] Ajouter support multi-repos
- [ ] Implémenter alerting (Slack/Email)
- [ ] Déploiement Kubernetes
- [ ] API REST pour les prédictions
- [ ] Tests unitaires et intégration
- [ ] CI/CD avec GitHub Actions

---

## 👥 Contributeurs

| Nom              | Rôle                   |
|------------------|------------------------|
| Yahya BAHLOUL    | Etudiant               |
| Zakaria BOUGAYOU | Etudiant               |

---

## 🙏 Remerciements

- [Apache Spark](https://spark.apache.org/)
- [Apache Kafka](https://kafka.apache.org/)
- [MongoDB](https://www.mongodb.com/)
- [MLflow](https://mlflow.org/)
- [Streamlit](https://streamlit.io/)

---

<p align="center">
  <b>⭐ Si ce projet vous aide, n'hésitez pas à lui donner une étoile !</b>
</p>