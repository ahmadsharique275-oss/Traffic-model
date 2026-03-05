import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Page configuration
st.set_page_config(page_title="Traffic Sign AI", page_icon="🚦", layout="centered")

st.title("🚦 Traffic Sign Recognition System")
st.write("Upload a traffic sign image to detect it.")

# Load model only once
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload Traffic Sign Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Traffic Sign"):

        with st.spinner("Analyzing Image..."):

            img_array = np.array(image)

            results = model.predict(img_array)

            res_plotted = results[0].plot()

            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            st.image(res_rgb, caption="Detection Result", use_container_width=True)

            st.subheader("Detected Signs")

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])

                    st.success(f"{class_name.upper()}  | Confidence: {confidence:.2f}")
