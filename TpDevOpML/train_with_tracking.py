"""
Régression des scores d'examen - Avec MLflow Tracking
Le scientifique organisé qui documente tout ! 🔬📋
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

print("🔬 Expérience ML (régression) avec MLflow")
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 1. Charger les données
df = pd.read_csv("student_habits_performance.csv")
print("📁 Données chargées.")

# 2. Préparation des features
X = df.drop(columns=["exam_score", "student_id"])
y = df["exam_score"]

# Colonnes catégorielles à encoder
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

# 3. Pipeline de prétraitement
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder='passthrough')

# 4. Pipeline complet
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ))
])

# 5. Division train/test
test_size = 0.2
random_seed = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed
)

# 🎯 Lancer une expérience MLflow
mlflow.set_experiment("student-score-regression-project")

with mlflow.start_run():
    print("🚀 Nouvelle run MLflow démarrée")

    # 🔧 Log des paramètres
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_seed", random_seed)
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("categorical_cols", ", ".join(categorical_cols))

    # 👨‍🏫 Entraînement
    model.fit(X_train, y_train)

    # 🔍 Prédictions
    y_pred = model.predict(X_test)

    # 📈 Évaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    # 📊 Log des métriques
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("train_samples", len(X_train))
    mlflow.log_metric("test_samples", len(X_test))

    # Create model signature and example
    signature = infer_signature(X_train, y_train)
    input_example = X_train.iloc[:5]  # Use first 5 rows as example

    # 💾 Sauvegarde du modèle
    mlflow.sklearn.log_model(
        model, 
        name="student_model",  # Using name instead of artifact_path
        signature=signature,
        input_example=input_example
    )

    # ✅ Résumé
    print(f"✅ Modèle entraîné avec R2: {r2:.2%}, RMSE: {rmse:.2f}")
    print("📋 Expérience enregistrée dans MLflow !")
