import streamlit as st
import requests
from PIL import Image

API_URL = "https://pneumonia-detection-api-8lh9.onrender.com/predict"

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁"
)

st.title("🫁 Pneumonia Detection System")

st.caption(
    "Developed by Azhar Khan | M.Tech Biomedical Engineering | IIT Indore"
)

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

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

        if result["prediction"] == "PNEUMONIA":

            st.markdown(
                f"""
                <div style="
                background-color:#ffebee;
                padding:20px;
                border-radius:10px;
                border-left:8px solid red;">
                <h2>🔴 PNEUMONIA</h2>
                <h3>Confidence: {result['confidence']}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div style="
                background-color:#e8f5e9;
                padding:20px;
                border-radius:10px;
                border-left:8px solid green;">
                <h2>🟢 NORMAL</h2>
                <h3>Confidence: {result['confidence']}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("---")
st.markdown(
    "**Disclaimer:** This tool is for educational and research purposes only and should not be used as a substitute for professional medical diagnosis."
)