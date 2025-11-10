import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image


model = load_model('best_model.pth')

# Emotion Labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit
st.title("Facial Emotion Recognition 😄")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # grayscale
    image = image.resize((48,48))
    img_array = np.array(image)/255.0
    img_array = img_array.reshape(1,48,48,1)
    
    prediction = model.predict(img_array)
    emotion = emotions[np.argmax(prediction)]
    
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write(f"Predicted Emotion: **{emotion}**")

