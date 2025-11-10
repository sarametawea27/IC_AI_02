import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load the PyTorch model
model = torch.load('best_model.pth', map_location=torch.device('cpu'))
model.eval()

# Emotion Labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit App
st.title("Facial Emotion Recognition 😄")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # grayscale
    transform = transforms.Compose([
        transforms.Resize((48,48)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)  # add batch dimension
    
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()
        emotion = emotions[prediction]
    
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write(f"Predicted Emotion: **{emotion}**")


