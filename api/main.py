from fastapi import FastAPI, Request
import pandas as pd
import joblib

# Charger le modèle au démarrage
model = joblib.load("model_lightGBM.pkl")

app = FastAPI(title="Home Credit API")

def use_model(data: Request):
    df = pd.DataFrame(data)
    # Charger le modèle depuis MLflow
    proba_list = model.predict_proba(df)
    preds_list = model.predict(df)
    accepted_loan = f"Le prêt peut être accordé (probabilité de non remboursement inférieure au seuil {model.best_threshold_})"
    rejected_loan = f"Le prêt ne peut pas être accordé (probabilité de non remboursement supérieure au seuil {model.best_threshold_})"
    verdict = [accepted_loan if pred == 0 else rejected_loan for pred in preds_list]
    # print(preds)
    return {"Verdict": str(verdict), "Probabilité de non remboursement": str(proba_list), "Seuil utilisé" : str(model.best_threshold_)}

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API Home Credit !"}

@app.post("/predict")
async def model_predict(request: Request):
    data = await request.json()

    use_model(data = data)