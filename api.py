from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Pneumonia Detection API")

# Load model once
model = tf.keras.models.load_model(
    "models/pneumonia_mobilenet.keras",
    compile=False
)

def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    return img

@app.get("/")
def home():
    return {
        "message": "Pneumonia Detection API Running"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents))

    x = preprocess(image)

    pred = float(model.predict(x)[0][0])

    label = "PNEUMONIA" if pred > 0.5 else "NORMAL"

    confidence = pred if pred > 0.5 else (1 - pred)

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    }