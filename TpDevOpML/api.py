"""
API FastAPI pour le modèle de prédiction des performances étudiantes
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import uvicorn

# Création de l'application FastAPI
app = FastAPI(
    title="API Prédiction Performance Étudiante",
    description="API pour prédire les performances des étudiants basée sur leurs habitudes",
    version="1.0.0"
)

# Modèle Pydantic pour la validation des données d'entrée
class StudentFeatures(BaseModel):
    age: int
    gender: str
    study_hours_per_day: float
    social_media_hours: float
    netflix_hours: float
    part_time_job: str
    attendance_percentage: float
    sleep_hours: float
    diet_quality: str
    exercise_frequency: int
    parental_education_level: str
    internet_quality: str
    mental_health_rating: int
    extracurricular_participation: str

# Charger le modèle au démarrage
@app.on_event("startup")
async def load_model():
    global model
    try:
        # Charger le modèle depuis MLflow
        model_name = "student-score-regressor"
        stage = "None"  # Peut être 'None', 'Staging', 'Production'
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors du chargement du modèle")

# Route pour la page d'accueil
@app.get("/")
def read_root():
    return {
        "message": "API de Prédiction des Performances Étudiantes",
        "endpoints": {
            "/predict": "POST - Faire une prédiction",
            "/health": "GET - Vérifier l'état de l'API"
        }
    }

# Route pour vérifier l'état de l'API
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Route pour faire des prédictions
@app.post("/predict")
def predict(student: StudentFeatures):
    try:
        # Convertir les données d'entrée en DataFrame
        input_data = pd.DataFrame([student.dict()])
        
        # Faire la prédiction
        prediction = model.predict(input_data)[0]
        
        # Retourner le résultat
        return {
            "predicted_score": round(prediction, 2),
            "input_features": student.dict()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

# Pour exécuter l'API localement
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 