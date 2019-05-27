import numpy as np
from keras.models import load_model
import io
from keras.preprocessing.image import img_to_array
from PIL import Image
from keras.models import model_from_json

file_json = 'selfie_model.json'
h5_file = 'selfie_model-weights.h5'
#model = load_model("selfie_model.h5")
read_json = open(file_json, 'r')
loaded_model_json = read_json.read()
read_json.close()
model= model_from_json(loaded_model_json)
model.load_weights(h5_file)

image = Image.open('Positive-(1).jpg')
image = image.resize((256,256), Image.ANTIALIAS)
image = image.convert("L")

imgar = img_to_array(image)
imgar = np.expand_dims(imgar, axis=0)

pred = model.predict(imgar)
#classes = model.predict_classes(imgar)
#label = classes[0]
label = np.argmax(pred)
digit = np.max(pred)
if label == 0:
    name = "fix"
elif label == 1:
    name = "not"
else:
    name = "other"
prediction = {"prediction":{'label':label, 'probability':digit}, "class":name}

print(prediction)