import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Page Config
st.set_page_config(page_title="AI Traffic Assistant", page_icon="🚦")

st.title("🚦 Smart Traffic Sign Analyzer")
st.write("Upload a traffic sign, and the AI will explain its meaning and accuracy.")

# 1. Sign Meanings (Customize these labels based on your best.pt classes)
sign_explanations = {
    "stop": "This sign indicates that you must come to a complete STOP.",
    "turn right": "This sign tells you that you must TURN RIGHT ahead.",
    "turn left": "This sign tells you that you must TURN LEFT ahead.",
    "speed limit 30": "This sign indicates a maximum SPEED LIMIT of 30 km/h.",
    "no entry": "This sign means NO ENTRY for vehicles in this direction.",
    # Note: If your model names are different, change the keys above to match them.
}

# Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# File Upload
uploaded_file = st.file_uploader("Choose a traffic sign photo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Photo', use_container_width=True)
    
    st.markdown("---")
    results = model(image)
    
    detected = False
    for result in results:
        boxes = result.boxes
        for box in boxes:
            detected = True
            label = model.names[int(box.cls)].lower() # Get name from model
            confidence = float(box.conf) * 100
            
            # --- DISPLAY RESULTS ---
            st.subheader("📊 Analysis Results")
            
            # 1. Prediction Name
            st.markdown(f"### **Prediction:** {label.upper()}")
            
            # 2. What it is saying (The Meaning)
            meaning = sign_explanations.get(label, f"This sign is identified as {label.upper()}.")
            st.info(f"💡 **What it means:** {meaning}")
            
            # 3. Accuracy
            st.success(f"📈 **Accuracy (Confidence):** {confidence:.2f}%")
            
    if not detected:
        st.error("No traffic sign detected. Please upload a clearer image.")

st.markdown("---")
st.caption("Professional AI Recognition Project")
