"""
Chargement et prédiction avec un modèle MLflow (régression)
Adapté au dataset student_habits_performance.csv 📚
"""
import mlflow
import mlflow.sklearn
import pandas as pd

print("\n2️⃣ Chargement et utilisation du modèle sauvegardé...")

# Exemple de nouvel étudiant (les colonnes doivent correspondre au dataset original)
new_data = pd.DataFrame([{
    "age": 20,
    "gender": "Female",
    "study_hours_per_day": 4.5,
    "social_media_hours": 2.0,
    "netflix_hours": 1.5,
    "part_time_job": "No",
    "attendance_percentage": 85.0,
    "sleep_hours": 7.0,
    "diet_quality": "Good",
    "exercise_frequency": 3,
    "parental_education_level": "Bachelor",
    "internet_quality": "Good",
    "mental_health_rating": 8,
    "extracurricular_participation": "Yes"
}])

# 📦 Charger le modèle depuis MLflow
model_name = "student-score-regressor"
stage = "None"  # Can be 'None', 'Staging', 'Production'
loaded_model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")

# 🔮 Faire une prédiction
predicted_score = loaded_model.predict(new_data)[0]

print("✅ Modèle chargé avec succès")
print(f"🔍 Prédiction du score d'examen : {predicted_score:.2f}/100")
