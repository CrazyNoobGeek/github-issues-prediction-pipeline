import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Configuration MLflow (On pointe vers le serveur Docker)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("GitHub_Issues_Classification")

print("🚀 Démarrage de l'entraînement MLOps...")

# Simulation de données (En attendant les vraies données de Mongo)
# Disons qu'on a vectorisé le texte (TF-IDF)
X_train = np.array([[0.1, 0.2], [0.4, 0.5], [0.1, 0.1], [0.9, 0.8]])
y_train = np.array([0, 1, 0, 1]) # 0=Bug, 1=Feature

# 2. Démarrage d'une "Run" MLflow
with mlflow.start_run():
    # Paramètres du modèle (Hyperparamètres)
    C_param = 0.5
    solver = 'lbfgs'
    
    # On loggue les paramètres pour s'en souvenir
    mlflow.log_param("C", C_param)
    mlflow.log_param("solver", solver)
    
    # Entraînement
    model = LogisticRegression(C=C_param, solver=solver)
    model.fit(X_train, y_train)
    
    # Évaluation (Simulation)
    accuracy = 0.85
    
    # On loggue la performance
    mlflow.log_metric("accuracy", accuracy)
    
    # 3. On sauvegarde le modèle DANS MLflow
    mlflow.sklearn.log_model(model, "model_logistic_regression")
    
    print(f"✅ Modèle entraîné avec succès ! Accuracy: {accuracy}")
    print("👉 Vérifie http://localhost:5000 pour voir les courbes.")