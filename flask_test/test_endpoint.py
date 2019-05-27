import requests

KERAS_REST_API_URL = "http://127.0.0.1:5000/predict_ktp"
IMAGE_PATH = "beagle-detail.jpg"

image = open(IMAGE_PATH, "rb").read()
payload = {"file": image}
r = requests.post(KERAS_REST_API_URL, files=payload).json()
print(r)

# the output
# {'prediction': {'class': 'not_ktp',
#   'label': 1,
#   'probability': 0.916510820388794}}