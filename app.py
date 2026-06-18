import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Pneumonia Grad-CAM", layout="centered")
st.title("🫁 Pneumonia Detection + Grad-CAM")

# -----------------------------
# LOAD MODEL (SAFE)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "models/pneumonia_model_v2.keras",
        compile=False
    )

    # 🔥 FORCE GRAPH BUILD (IMPORTANT)
    _ = model(tf.zeros((1, 224, 224, 3)))

    return model

model = load_model()
print("INPUTS =", model.inputs)


LAST_CONV_LAYER = "conv2d_2"

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------
# GRAD-CAM (STABLE METHOD)
# -----------------------------
def make_gradcam_heatmap(img_array, model, layer_name):

    last_conv_layer = model.get_layer(layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.outputs[0]
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    print("GRADS =", grads)

    if grads is None:
        raise ValueError("GRADS IS NONE")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()
# -----------------------------
# OVERLAY
# -----------------------------
def overlay_heatmap(img, heatmap):

    img = np.array(img.convert("RGB").resize((224, 224)))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# -----------------------------
# UI
# -----------------------------
file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if file:

    image = Image.open(file)
    st.image(image, caption="Original Image", use_container_width=True)

    x = preprocess(image)

    # -------------------------
    # Prediction
    # -------------------------
    pred = model.predict(x)[0][0]

    label = "PNEUMONIA ❌" if pred > 0.5 else "NORMAL ✅"

    st.subheader("Prediction")
    st.write(label)
    st.write("Confidence:", float(pred))

    # -------------------------
    # Grad-CAM
    # -------------------------
    st.subheader("Grad-CAM Heatmap")

    heatmap = make_gradcam_heatmap(x, model, LAST_CONV_LAYER)

    st.image(heatmap, caption="Heatmap")

    result = overlay_heatmap(image, heatmap)

    st.image(result, caption="Grad-CAM Overlay")

    st.success("Done 🚀")