from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# chemin de mon modèle MLflow
model = model_uri = f"runs:/{run_id}/student_model"

# Exemple : structure d’entrée attendue
class InputData(BaseModel):
    amount: float
    use_chip: bool

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    return {"result": prediction[0]}
