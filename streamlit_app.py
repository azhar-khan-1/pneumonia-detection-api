import streamlit as st
import requests
from PIL import Image

API_URL = "https://pneumonia-detection-api-8lh9.onrender.com/predict"

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁"
)

st.title("🫁 Pneumonia Detection")

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):

        with st.spinner("Analyzing..."):

            files = {
                "file": uploaded_file.getvalue()
            }

            response = requests.post(
                API_URL,
                files=files
            )

            result = response.json()

        st.success("Prediction Complete")

        st.write(
            f"### Prediction: {result['prediction']}"
        )

        st.write(
            f"### Confidence: {result['confidence']}%"
        )