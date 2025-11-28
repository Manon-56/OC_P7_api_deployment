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