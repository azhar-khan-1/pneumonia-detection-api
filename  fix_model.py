import tensorflow as tf

model = tf.keras.models.load_model(
    "models/pneumonia_model_v2.keras",
    compile=False
)

print("Model loaded successfully")

# 🔥 re-save in stable format
model.save("models/pneumonia_model_FIXED.keras")

print("Model re-saved successfully")