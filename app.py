from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import tensorflow as tf
import tensorflow.keras as keras
import json
import numpy as np


desired_width = 100
desired_height = 75
labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

with open('case_treatment_pl.json', 'r', encoding='utf-8') as file:
    json_data = file.read()

info = json.loads(json_data)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

model = keras.models.load_model('./saved_conv_next')


@app.get("/")
async def home():
    return {"Hello": "doctor"}


@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    resized_image = image.resize((desired_width, desired_height)) 
    
    img_array = keras.preprocessing.image.img_to_array(resized_image)
    img_array = tf.expand_dims(img_array, 0) 

    # predictions = [[0.6,  0, 0.2, 0.2, 0, 0, 0]] # for faster tests
    predictions = model.predict(img_array)

    prediction_class = np.argmax(predictions, axis=1)[0]
    print(info[labels[prediction_class]])

    return info[labels[prediction_class]]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)