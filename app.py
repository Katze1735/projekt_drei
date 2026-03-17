import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

# Load YOLOv8 World v2 model (will download automatically if not present)
@st.cache_resource
def load_model():
    model = YOLO("yolov8x-worldv2.pt")
    return model

model = load_model()

st.title("YOLOv8 World v2 Object Detection App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy array
    image_np = np.array(image)

    # Run inference
    results = model(image_np)

    # Get annotated image
    annotated_frame = results[0].plot()

    # Convert BGR to RGB for Streamlit display
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    st.image(annotated_frame, caption="Detected Objects", use_column_width=True)

    # Display detected objects
    st.subheader("Detected Objects")

    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            st.write(f"{class_name}: {conf:.2f}")
    else:
        st.write("No objects detected.")

st.markdown("---")
st.markdown("Built with Streamlit and YOLOv8 World v2")
