#import module for make endpoint
from flask import Flask, jsonify, request

#import module for load model and testing image using model
from keras.models import load_model, model_from_json
#from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

#import module for image preprocessing
import numpy as np
import io
from PIL import Image


app = Flask(__name__) #create an instance of the Flask class 

#load model for selfie classification
selfie_file_json = 'selfie_model.json'
selfie_file_h5 = 'selfie_model-weights.h5'
#model = load_model("selfie_model.h5")
selfie_read_json = open(selfie_file_json, 'r')
selfie_loaded_model = selfie_read_json.read()
selfie_read_json.close()
selfie_model= model_from_json(selfie_loaded_model)
selfie_model.load_weights(selfie_file_h5)

#load model for ktp classification
ktp_file_json = 'ktp_model.json'
ktp_file_h5 = 'ktp_model-weights.h5'
#model = load_model("selfie_model.h5")
ktp_read_json = open(ktp_file_json, 'r')
ktp_loaded_model = ktp_read_json.read()
ktp_read_json.close()
ktp_model= model_from_json(ktp_loaded_model)
ktp_model.load_weights(ktp_file_h5)


#define function for selfie classification endpoint
@app.route('/predict_selfie', methods=["POST"])
def predict_images_selfie():
    image = request.files['file'].read() #request image file
    image = Image.open(io.BytesIO(image))
    image = image.resize((256,256), Image.ANTIALIAS) # because the model using size (256,256), so all image which to predict must resize into 256,256
    image = image.convert("L") #convert image from RGB to Grayscale
    images = img_to_array(image) # Image into array
    images = images(matrix.reshape(1, *matrix.shape)) #image array = (256,256,1) to image array = (1, 256, 256, 1)
    
    pred = selfie_model.predict(images) #predict image
    #prediction = {'digit':int(digit)}
    label = np.argmax(pred) #to take the maximum probability from all class
    digit = np.max(pred) #to show the maximum probability
    if label == 0:
        name = "fix_selfie"
    elif label == 1:
        name = "not_selfie"
    else:
        name = "blurred_selfie"
    prediction = {'prediction':{'class':int(label), 'label':str(name), 'probability':float(digit)}} #define the output
    return jsonify(prediction)

#define function for selfie classification endpoint
@app.route('/predict_ktp', methods=["POST"])
def predict_images_ktp():
    image = request.files['file'].read()
    image = Image.open(io.BytesIO(image))
    image = image.resize((256,256), Image.ANTIALIAS)
    image = image.convert("L")
    images = img_to_array(image)
    images = images(matrix.reshape(1, *matrix.shape))
    
    pred = selfie_model.predict(images)
    #digit = np.argmax(pred)
    #prediction = {'digit':int(digit)}
    label = np.argmax(pred)
    digit = np.max(pred)
    if label == 0:
        name = "fix_ktp"
    elif label == 1:
        name = "not_ktp"
    else:
        name = "blurred_ktp"
    prediction = {'prediction':{'class':int(label), 'label':str(name), 'probability':float(digit)}}
    return jsonify(prediction)

if __name__ == "__main__":
    app.run()