"""Contains test methods checking FastAPI endpoints responses."""

import requests
import json
import ast
from api.main import use_model

def test_model_output():
    """
        Check the output of the model is as expected : 
        - the probability is contained between 0 and 1,
        - the used threshold is contaiend between 0 and 1,
        - the verdict is coherent with the threshold and the probability
    """

    # Opening JSON file
    with open('requete.json') as json_file:
        requete = json.load(json_file)
    
    response = use_model(requete)

    proba_list = ast.literal_eval(response.get('Probabilité de remboursement'))
    assert all(((proba <1) and (proba >0)) for proba in  proba_list)
    seuil = ast.literal_eval(response.get('Seuil utilisé'))
    assert (seuil>0) and (seuil<1)
    verdict_list = response.get("Verdict")
    for idx, verdict in enumerate(verdict_list):
        if verdict.startswith("Le prêt peut être accordé"):
            assert proba_list[idx]<seuil
        elif verdict.startswith("Le prêt ne peut pas être accordé"):
            assert proba_list[idx]<seuil
        else:
            print("There is an error")