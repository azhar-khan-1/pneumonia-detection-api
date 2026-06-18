import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Original model load
old_model = tf.keras.models.load_model(
    "models/pneumonia_model_v2.keras",
    compile=False
)

# Rebuild architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

# Build model
_ = model(tf.zeros((1,224,224,3)))
print("MODEL OUTPUTS =", model.outputs)

# Copy weights
model.set_weights(old_model.get_weights())

print("REBUILT MODEL READY")
print(model.summary())
import numpy as np
from PIL import Image

# Test image
img = Image.open(
    "dataset/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
).convert("RGB")

img = img.resize((224, 224))

img_array = np.array(img).astype(np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Feature extractor
grad_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=[
        model.get_layer("conv2d_2").output,
        model.outputs[0]
    ]
)

x = tf.convert_to_tensor(img_array, dtype=tf.float32)

with tf.GradientTape() as tape:

    conv_outputs, preds = grad_model(x)

    class_channel = preds[:, 0]

grads = tape.gradient(class_channel, conv_outputs)

print("GRADS IS NONE =", grads is None)

if grads is not None:

    print("GRADS SHAPE =", grads.shape)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    heatmap = heatmap.numpy()

    print("HEATMAP SHAPE =", heatmap.shape)
    print("HEATMAP MIN =", heatmap.min())
    print("HEATMAP MAX =", heatmap.max())
    import cv2

heatmap_uint8 = np.uint8(255 * heatmap)

heatmap_resized = cv2.resize(
    heatmap_uint8,
    (224, 224)
)

cv2.imwrite(
    "heatmap.jpg",
    heatmap_resized
)

print("HEATMAP SAVED")
heatmap_color = cv2.applyColorMap(
    heatmap_resized,
    cv2.COLORMAP_JET
)

cv2.imwrite(
    "heatmap_color.jpg",
    heatmap_color
)

print("COLOR HEATMAP SAVED")
original = cv2.cvtColor(
    np.array(img),
    cv2.COLOR_RGB2BGR
)

original = cv2.resize(original, (224,224))

overlay = cv2.addWeighted(
    original,
    0.6,
    heatmap_color,
    0.4,
    0
)

cv2.imwrite(
    "gradcam_overlay.jpg",
    overlay
)

print("OVERLAY SAVED")