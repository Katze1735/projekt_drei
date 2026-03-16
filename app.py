import streamlit as st
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Clothes Segmentation AI", layout="wide")

st.title("👕 Clothes Detection with SegFormer")
st.write("Upload an image and the AI will identify clothing regions.")

@st.cache_resource
def load_model():
    processor = SegformerImageProcessor.from_pretrained(
        "mattmdjaga/segformer_b2_clothes"
    )
    model = AutoModelForSemanticSegmentation.from_pretrained(
        "mattmdjaga/segformer_b2_clothes"
    )
    return processor, model

processor, model = load_model()

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Original Image", use_column_width=True)

    with st.spinner("Analyzing clothes..."):

        inputs = processor(images=image, return_tensors="pt")

        outputs = model(**inputs)

        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.imshow(pred_seg, alpha=0.5)
        ax.axis("off")

        st.pyplot(fig)

        st.success("Clothing segmentation complete!")
