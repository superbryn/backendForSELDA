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

@app.route("/predict", methods=['POST'])
def prediction():
    #get the base64 image from the request
    data = request.json['image']

    
    imageData = base64.b64decode(data) #decode the base64 into binary (image)
    img = Image.open(io.BytesIO(imageData))
    img = np.array(img)
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

        return jsonify({'prediction': prediction_character}) # prediction

    return jsonify({'prediction': 'No Hands Found'}) #if no hands are found

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2000, debug=True)
