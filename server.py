from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import mediapipe as mp
import pickle as achaar
import base64
import io
from PIL import Image
import cv2 as cv

#initializing the app
app = Flask(__name__)
CORS(app)
mp_hands = mp.solutions.hands 
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

modelDict = achaar.load(open('./newmodel.p', 'rb'))
model = modelDict['model']

labelDict = {chr(i): chr(i) for i in range(65, 91)}
#i used to write down this entire hash from A to Z and one day i realized i could use this method instead of doin that boring ass stuff i did backthen
#so basically label data is a dictionary which works a place holder. this is something like a hashmap a hashmap

@app.route("/predict", methods=['POST'])
def prediction(): #the main function which predicts the model
    #get the base64 image from the request
    data = request.json['image']

    #decode the image
    #for people who don't know much about cs Base64 is an encoding method used to convert binary data which is 
    #for us images are in text format using ASCII (American standard code for information interchange) and then we decode that run that
    imageData = base64.b64decode(data)
    img = Image.open(io.BytesIO(imageData))
    img = np.array(img) # this is turned into an np array so that the dimensions of the image could me obtained without doin some complicated stuff
                        # the original model used asarray

    #imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB) this line was a pain in the ass for me as the normal model used this i thought this was useful
    #i leave this line as a souvenier for all my friend who reviews this code and reaslizes what an idiot i am LoL XD
    result = hands.process(img)
    
    dataAUX = []
    x_ = []
    y_ = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                dataAUX.append(landmark.x - min(x_))
                dataAUX.append(landmark.y - min(y_))

        prediction = model.predict([dataAUX])  
        prediction_character = labelDict.get(prediction[0], "Unknown")

        return jsonify({'prediction': prediction_character})

    return jsonify({'prediction': 'no hands'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2000, debug=True)
