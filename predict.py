from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Model load
model = load_model("models/pneumonia_model.keras")

# Test image path
img_path = "dataset/chest_xray/test/NORMAL/IM-0023-0001.jpeg"

# Image preprocessing
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Prediction: PNEUMONIA")
    print("Confidence:", prediction[0][0]*100)
else:
    print("Prediction: NORMAL")
    print("Confidence:", (1-prediction[0][0])*100)