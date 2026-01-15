import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦", layout="centered")

st.title("🚦 Traffic Sign Recognition System")
st.write("Upload a photo (even from a mobile screen), and AI will identify the sign and explain the rule.")
st.markdown("---")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# --- 3. MEANING DICTIONARY (English Explanations) ---
def get_sign_meaning(label):
    label = label.lower()
    if "stop" in label:
        return "⚠️ STOP: You must bring your vehicle to a complete halt."
    elif "speed" in label:
        return "⚡ SPEED LIMIT: Do not exceed the speed limit displayed."
    elif "left" in label:
        return "⬅️ TURN LEFT: You must turn left ahead."
    elif "right" in label:
        return "➡️ TURN RIGHT: You must turn right ahead."
    elif "no" in label and "entry" in label:
        return "⛔ NO ENTRY: Vehicles are not allowed to enter."
    elif "yield" in label:
        return "⚠️ YIELD: Give way to other vehicles."
    elif "traffic" in label or "signal" in label:
        return "🚦 TRAFFIC SIGNAL: Follow the traffic light colors."
    elif "pedestrian" in label:
        return "🚶 PEDESTRIAN CROSSING: Slow down and watch for people crossing."
    else:
        return "ℹ️ INFO: Follow the indicated traffic regulation."

# --- 4. INPUT SECTION ---
uploaded_file = st.file_uploader("Upload Traffic Sign Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # A. Show User Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Input', use_container_width=True)
    
    st.markdown("### 🔍 AI Analysis Report")
    
    # B. Processing
    with st.spinner('AI is scanning the image...'):
        results = model(image)
        
        # C. Draw Boxes (Like your screenshot)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB) # Colors fix
        st.image(res_rgb, caption='AI Detection Result', use_container_width=True)
        
        # D. List All Detections
        detected = False
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detected = True
                
                # Data Extraction
                label = model.names[int(box.cls)]
                confidence = float(box.conf) * 100
                meaning = get_sign_meaning(label)
                
                # E. Final Output (Card Style)
                with st.expander(f"Detected: {label.upper()} ({confidence:.1f}%)", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sign Name", label.upper())
                    with col2:
                        st.metric("Accuracy", f"{confidence:.2f}%")
                    
                    st.info(f"**Action Required:** {meaning}")

        if not detected:
            st.warning("No traffic signs were detected in this image.")

# Footer
st.markdown("---")
st.caption("AI Traffic Assistant | Powered by YOLOv8")
