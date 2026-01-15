import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🚦 Traffic Sign Recognition System")
st.write("A deep learning application to identify traffic signs with high precision.")
st.markdown("---")

# --- MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    model_path = 'best.pt'
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.error(f"Error: '{model_path}' not found in the repository. Please ensure your model file is uploaded.")
        return None

model = load_trained_model()

# --- FILE UPLOAD SECTION ---
uploaded_file = st.file_uploader("Upload a traffic sign image for analysis", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.markdown("### Analysis Report")
    
    if model is not None:
        with st.spinner('Analyzing the image...'):
            # Run Inference
            results = model(image)
            
            detected = False
            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    detected = True
                    for box in boxes:
                        label = model.names[int(box.cls)]
                        confidence = float(box.conf) * 100
                        
                        # --- PROFESSIONAL DISPLAY ---
                        st.success(f"### Prediction: **{label.upper()}**")
                        st.info(f"### Accuracy (Confidence): **{confidence:.2f}%**")
                
            if not detected:
                st.warning("No traffic signs detected. Please try a clearer or closer image.")
    else:
        st.error("Model failed to load. Please check your GitHub files.")

# --- FOOTER ---
st.markdown("---")
st.caption("AI-Powered Traffic Analysis Project | Powered by YOLOv8")
