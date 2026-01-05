import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.h5")

model = load_model()

st.title("🧠 Brain Tumor Detection System (AI Powered)")
st.write("Upload a brain MRI image to detect Tumor.")

def preprocess_image(image):
    image = image.resize((224,224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("Detect Tumor"):
        with st.spinner("Analyzing MRI..."):
            img = preprocess_image(image)
            prediction = model.predict(img)[0][0]

            if prediction >= 0.5:
                st.error("🧠 Tumor Detected")
                st.write(f"Confidence: {prediction*100:.2f}%")
            else:
                st.success("✅ No Tumor Detected")
                st.write(f"Confidence: {(1-prediction)*100:.2f}%")
