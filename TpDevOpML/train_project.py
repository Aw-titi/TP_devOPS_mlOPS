"""
Script d'entraînement avec paramètres en ligne de commande
Basé sur le dataset student_habits_performance.csv
"""

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature

def main():
    # Récupérer les paramètres passés au script
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="student_habits_performance.csv")
    args = parser.parse_args()

    # Lire le fichier CSV
    df = pd.read_csv(args.data_path)
    X = df.drop(columns=["exam_score", "student_id"])
    y = df["exam_score"]

    # Trouver les colonnes catégorielles
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    # Préparation du préprocesseur pour encoder ces colonnes
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ], remainder="passthrough")

    # Construire le pipeline complet avec modèle RandomForest
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_seed
        ))
    ])

    # Séparer les données en train et test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed
    )

    # Lancer un run MLflow pour tracer tout ça
    mlflow.set_experiment("student-score-regression-project")
    with mlflow.start_run():
        # Enregistrer les paramètres pour garder une trace
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_seed", args.random_seed)
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("categorical_cols", ", ".join(categorical_cols))

        # Entraîner le modèle
        pipeline.fit(X_train, y_train)

        # Prédire sur le jeu de test
        y_pred = pipeline.predict(X_test)

        # Calculer les erreurs et score
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        # Enregistrer les métriques
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Préparer la signature du modèle et un exemple d'entrée
        signature = infer_signature(X_train, y_train)
        input_example = X_train.iloc[:5]

        # Sauvegarder le pipeline complet dans MLflow
        mlflow.sklearn.log_model(
            pipeline,
            name="student_model",
            signature=signature,
            input_example=input_example
        )

        print(f"[OK] Entraînement terminé - R²: {r2:.2%}, RMSE: {rmse:.2f}")

if __name__ == "__main__":
    main()
