import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

st.set_page_config(page_title="Clothing Identifier", layout="centered")

st.title("👕 Clothing Item Identifier")
st.write("Upload an image and the AI will identify the clothing item.")

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(
        "wargoninnovation/wargon-clothing-classifier"
    )
    model = AutoModelForImageClassification.from_pretrained(
        "wargoninnovation/wargon-clothing-classifier"
    )
    return processor, model

processor, model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing clothing..."):

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class_id = logits.argmax(-1).item()

        label = model.config.id2label[predicted_class_id]

    st.success(f"Detected clothing item: **{label}**")
