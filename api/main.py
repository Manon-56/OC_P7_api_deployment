from fastapi import FastAPI, Request
import pandas as pd
import joblib
import pickle as pkl

class ModelWithThreshold:
    """
    Classe permettant d'encapsuler le modèle choisi avec le threshold optimal associé et donc d'utiliser directement la méthode predict et d'enregistrer le threshold au sein même du modèle sans devoir passer par deux fichiers séparés. 
    """
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
    
    def predict(self, X):
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)


# Charger le modèle au démarrage
with open("model_lightGBM.pkl", 'r') as file:
    model = pkl.load(file)

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