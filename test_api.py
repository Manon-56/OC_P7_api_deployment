"""Contains test methods checking FastAPI endpoints responses."""

import requests
import json
import ast

def test_predict_endpoint():
    """Check connection with FastAPI predict endpoint.

    """

    # Use azure endpoint
    BASE_URL = "https://api-oc-p7-a7gwfebjc6csbpe4.francecentral-01.azurewebsites.net/"
    # importing the module


    # Opening JSON file
    with open('requete.json') as json_file:
        requete = json.load(json_file)
    
    response = requests.post(f"{BASE_URL}/predict", json=requete)

    # Check response status
    assert response.status_code == 200
    

    # Check response data structure
    prediction = ast.literal_eval(response.json().get('Prédiction'))
    assert isinstance(
        prediction, list
    )  # check response global format
    
    pass