"""
Projet : prédire les scores d'examen des étudiants
Objectif : tester un modèle de régression avec suivi via MLflow
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime
from mlflow.models.signature import infer_signature

print("Début de l'expérience")
print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Chargement du fichier CSV contenant les données des étudiants
df = pd.read_csv("student_habits_performance.csv")
print("Données chargées.")

# On enlève l'ID et la cible, le reste servira de features
X = df.drop(columns=["exam_score", "student_id"])
y = df["exam_score"]

# On repère les colonnes qui contiennent du texte (catégorielles)
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

# On prépare le traitement des colonnes catégorielles
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder='passthrough')

# Pipeline : d'abord le prétraitement, puis la régression
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ))
])

# On sépare les données en un jeu pour l'entraînement et un autre pour le test
test_size = 0.2
random_seed = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed
)

# On lance une nouvelle expérience MLflow (ou on rejoint une existante)
mlflow.set_experiment("student-score-regression-project")

with mlflow.start_run():
    print("Nouvelle run lancée dans MLflow")

    # On garde une trace des paramètres utilisés
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_seed", random_seed)
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("categorical_cols", ", ".join(categorical_cols))

    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Prédictions sur les données de test
    y_pred = model.predict(X_test)

    # Calcul des scores pour voir si le modèle s’en sort bien
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    # On enregistre les résultats dans MLflow
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("train_samples", len(X_train))
    mlflow.log_metric("test_samples", len(X_test))

    # Signature du modèle (utile pour la reproductibilité et les API)
    signature = infer_signature(X_train, y_train)
    input_example = X_train.iloc[:5]

    # Enregistrement du modèle dans MLflow
    mlflow.sklearn.log_model(
        model, 
        name="student_model",
        signature=signature,
        input_example=input_example
    )

    # Fin de run : on affiche les résultats principaux
    print(f"Modèle entraîné. Score R2 : {r2:.2%}, RMSE : {rmse:.2f}")
    print("Run terminée et sauvegardée dans MLflow.")
