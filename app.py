import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- Load Model ---
model = load_model("emotion_model.h5")

# Emotion labels
emotion_dict = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Streamlit UI ---
st.title("😊 Facial Emotion Recognition App")
st.write("Upload an image and detect emotions using AI 🎭")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert image to array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Process each face
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.reshape(1, 48, 48, 1) / 255.0

        prediction = model.predict(roi)
        emotion_label = emotion_dict[np.argmax(prediction)]

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    # Convert BGR → RGB for Streamlit
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", caption="Detected Emotions")

else:
    st.info("Please upload an image to analyze.")
