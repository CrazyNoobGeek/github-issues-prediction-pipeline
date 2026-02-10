# Utilisation d'une image Python légère
FROM python:3.9-slim

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système (nécessaire pour certains calculs)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers requis
COPY requirements.txt .

# Installation des librairies Python
RUN pip3 install -r requirements.txt && pip3 install plotly

# Copie du code de l'application
COPY . .

# Exposition du port Streamlit (par défaut 8501)
EXPOSE 8501

# Vérification de santé (Healthcheck)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Commande de lancement
#ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# On expose les deux ports potentiels

EXPOSE 8502

# Utilise CMD au lieu de ENTRYPOINT pour permettre l'écrasement par docker-compose
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]