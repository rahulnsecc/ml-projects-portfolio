import requests

data = {'text': 'Supply chain software encountering errors during restart process after RAM upgrade'}

response = requests.post('http://localhost:5000/predict', json=data)
print(response.json())
