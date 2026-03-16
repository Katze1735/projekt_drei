import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Clothing Detection", layout="centered")

st.title("👕 Clothing Detection with YOLO")
st.write("Upload an image to detect clothing items.")

# Load model
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # your clothing detection model
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Clothing"):

        with st.spinner("Running YOLO detection..."):

            results = model(image)

            result_image = results[0].plot()

            st.image(result_image, caption="Detected Clothing", use_column_width=True)

            boxes = results[0].boxes
            names = model.names

            st.subheader("Detected items:")

            detected = set()

            for box in boxes:
                class_id = int(box.cls)
                detected.add(names[class_id])

            for item in detected:
                st.write(f"• {item}")
