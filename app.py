import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Traffic Sign AI", page_icon="🚦", layout="centered")

st.title("🚦 Traffic Sign Recognition System")
st.markdown("### AI Traffic Assistant (Pro Version)")
st.write("Upload a traffic sign image. Use the slider to adjust sensitivity.")

# --- 2. SIDEBAR FOR ADJUSTMENTS (Ye naya feature hai) ---
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Model Sensitivity (Confidence)", 0.0, 1.0, 0.25, 0.05)
st.sidebar.info("Tip: If the correct sign is not showing, lower the sensitivity slider.")

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# --- 4. MEANING DICTIONARY ---
def get_sign_meaning(label):
    label = label.lower()
    if "stop" in label:
        return "🛑 STOP: You must come to a complete halt."
    elif "left" in label:
        return "⬅️ TURN LEFT: Turn left ahead."
    elif "right" in label:
        return "➡️ TURN RIGHT: Turn right ahead."
    elif "speed" in label or "limit" in label or "80" in label or "50" in label:
        return "⚡ SPEED LIMIT: Do not exceed the displayed speed limit."
    elif "no" in label and "entry" in label:
        return "⛔ NO ENTRY: Do not enter this road."
    elif "traffic" in label or "signal" in label:
        return "🚦 TRAFFIC SIGNAL: Follow the traffic lights."
    else:
        return f"ℹ️ INFO: Follow the rules for '{label.upper()}'."

# --- 5. MAIN APP ---
uploaded_file = st.file_uploader("Upload Traffic Sign Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_container_width=True)
    
    st.markdown("---")
    
    # Run Prediction with Custom Confidence
    results = model.predict(image, conf=confidence_threshold)
    
    # Display AI View
    res_plotted = results[0].plot()
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    st.image(res_rgb, caption='AI Detection View', use_container_width=True)
    
    # Process Results
    detected = False
    st.markdown("### 📢 Detection Results:")
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            detected = True
            label = model.names[int(box.cls)]
            confidence = float(box.conf) * 100
            meaning = get_sign_meaning(label)
            
            # Show Result Card
            with st.container():
                st.markdown(f"#### ✅ Detected: **{label.upper()}**")
                st.progress(int(confidence))
                st.write(f"**Accuracy:** {confidence:.2f}%")
                st.info(f"**Meaning:** {meaning}")
                st.markdown("---")
                
    if not detected:
        st.warning(f"No signs detected above {confidence_threshold*100}% accuracy. Try lowering the Sensitivity slider in the sidebar.")
