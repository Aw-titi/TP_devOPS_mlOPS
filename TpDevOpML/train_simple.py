"""
Régression des scores d'examen - Méthode classique (sans MLflow)
Le scientifique désorganisé qui ne note rien ! 📚🧪
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from datetime import datetime

print("🔬 Expérience ML classique - Méthode 'artisanale'")
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Paramètres d'entraînement (non tracés ailleurs !)
n_estimators = 15
max_depth = 3
test_size = 0.3
random_seed = 42

print(f"🧪 Test avec {n_estimators} arbres, profondeur {max_depth}, split {test_size}")

# 1. Chargement des données
df = pd.read_csv("student_habits_performance.csv")

target = "exam_score"
categorical_cols = ["study_hours_per_day", "sleep_hours"]
numeric_cols = ["attendance_percentage", "age", "social_media_hours"]

X = df[categorical_cols + numeric_cols]
y = df[target]

print(f"📊 Données: {len(X)} échantillons, {len(X.columns)} caractéristiques")

# 2. Prétraitement
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

# 3. Pipeline complet
model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_seed
    ))
])

# 4. Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed
)

# 5. Entraînement
model.fit(X_train, y_train)

# 6. Évaluation
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"📉 MSE: {mse:.2f}")
print(f"📈 R²: {r2:.2%}")

# 7. Sauvegarde "artisanale"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f"student_model_{timestamp}.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print(f"💾 Modèle sauvegardé dans: {model_filename}")
