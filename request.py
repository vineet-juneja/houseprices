import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'area': 3000, 'bedroom': 2, 'age': 15})

print(r.json())
