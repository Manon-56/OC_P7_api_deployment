from fastapi import FastAPI, Request
import pandas as pd
import joblib

# Charger le modèle au démarrage
model = joblib.load("model.pkl")

app = FastAPI(title="Home Credit API")

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API Home Credit !"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    df = pd.DataFrame(data) 

    # Charger le modèle depuis MLflow
    preds = model.predict(df)
    print(preds)
    return {"Prédiction": str(preds)}