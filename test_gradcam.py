import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model(
    "models/pneumonia_model_v2.keras",
    compile=False
)

# Force build
_ = model(tf.zeros((1, 224, 224, 3)))

print("MODEL LOADED")
print("Built =", model.built)
print("Inputs =", model.inputs)
print("Outputs =", model.outputs)

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.__class__.__name__)

# Load image
img = Image.open(
    "dataset/chest_xray/test/NORMAL/person1_virus_6.jpeg"
).convert("RGB")

img = img.resize((224, 224))

img_array = np.array(img).astype(np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
pred = model.predict(img_array)

print("PREDICTION =", pred)

# Conv layer
last_conv = model.get_layer("conv2d_1")

print("LAST CONV =", last_conv)


# NEW TEST

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer("conv2d_2").output
)

with tf.GradientTape() as tape:

    x = tf.convert_to_tensor(img_array, dtype=tf.float32)

    features = feature_extractor(x)

    tape.watch(features)

    preds = model(x, training=False)

    loss = preds[:, 0]

grads = tape.gradient(loss, features)

print("FEATURES SHAPE =", features.shape)
print("GRADS =", grads)