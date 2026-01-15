import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Simple Title
st.title("Traffic Sign Recognition")

# Load your model
model = YOLO('best.pt')

# Upload Button
uploaded_file = st.file_uploader("Upload a traffic sign photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Photo', use_container_width=True)
    
    st.write("Processing...")
    
    # Predict
    results = model(image)
    
    # Result display
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            st.write("No traffic sign detected.")
        for box in boxes:
            label = model.names[int(box.cls)]
            confidence = float(box.conf)
            # Simple result message
            st.success(f"Result: {label} (Confidence: {confidence:.2f})")
