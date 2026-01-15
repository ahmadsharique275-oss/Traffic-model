import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Traffic Sign AI", page_icon="🚦")
st.title("🚦 Traffic Sign Recognition System")
st.write("Upload a photo and the AI will identify it!")

# 2. Loading the Model
model = YOLO('best.pt')

# 3. Upload Button
file = st.file_uploader("Upload Traffic Sign Photo", type=['jpg', 'png', 'jpeg'])

if file is not None:
    img = Image.open(file)
    st.image(img, caption='Uploaded Photo', use_container_width=True)

    # 4. AI Prediction
    with st.spinner('AI is analyzing...'):
        results = model(img)
        res_plotted = results[0].plot()
        
        # Displaying the result image with detection boxes
        # Note: Agar photo neeli (Blue) dikhe, to niche wali line me channels="BGR" jod dena
        st.image(res_plotted, caption='AI Result', use_container_width=True, channels="BGR")
        
        # Success message
        st.success("Identification Complete!")
