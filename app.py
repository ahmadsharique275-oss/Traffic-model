import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. Page Setup
st.set_page_config(page_title="AI Traffic Assistant", page_icon="🚦")
st.title("🚦 Professional Traffic Sign Analyzer")
st.write("Upload a traffic sign image to get a detailed AI report.")

# 2. Load the Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 3. File Upload
uploaded_file = st.file_uploader("Upload Image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔍 AI Prediction & Analysis")
    
    # 4. Perform Detection
    results = model(image)
    
    detected = False
    for result in results:
        boxes = result.boxes
        for box in boxes:
            detected = True
            # Get detection data
            label = model.names[int(box.cls)]
            confidence = float(box.conf) * 100
            
            # --- DISPLAY BOX ---
            st.markdown(f"### **Prediction:** {label.upper()}")
            st.markdown(f"### **Accuracy:** {confidence:.2f}%")
            
            # --- WHAT IT IS SAYING (The Action) ---
            st.info(f"💡 **Action Required:** This traffic sign is identified as a **{label.upper()}** sign. You should follow the rules indicated by this signal for road safety.")

    if not detected:
        st.warning("No traffic sign detected. Please upload a clearer image of a single traffic sign.")

# Footer
st.markdown("---")
st.caption("AI-Powered Traffic Vision System | 2026 Professional Version")
