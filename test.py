import requests

url1 = "http://127.0.0.1:5000/predict"

data = {
    'bedrooms': 5,
    'bathrooms': 2,
    'sqft_living': 2500,
    'sqft_lot': 6000,
}

response = requests.post(url1, json=data)
print("Prediction Response:", response.json())