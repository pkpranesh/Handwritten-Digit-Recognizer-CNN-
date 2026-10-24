import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(page_title="Digit Recognizer", page_icon="✏️")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/mnist_cnn.h5')

model = load_model()

st.title("✏️ Handwritten Digit Recognizer")
st.markdown("### Upload a handwritten digit image — CNN predicts it!")
st.markdown("---")

uploaded = st.file_uploader("Upload digit image (28x28 or any size)", type=['png','jpg','jpeg'])

if uploaded:
    # Load and preprocess image
    img = Image.open(uploaded).convert('L')  # grayscale
    img = img.resize((28, 28))               # resize to 28x28
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Your Image", width=200)
    
    # Preprocess
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Predict
    predictions  = model.predict(img_array)
    predicted    = np.argmax(predictions)
    confidence   = predictions[0][predicted] * 100
    
    with col2:
        st.markdown(f"### Predicted Digit: {predicted}")
        st.markdown(f"### Confidence: {confidence:.1f}%")
        
        st.markdown("**All probabilities:**")
        for i, prob in enumerate(predictions[0]):
            st.progress(float(prob), text=f"Digit {i}: {prob*100:.1f}%")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "CNN Digit Recognizer | Built by Pranesh P K"
    "</p>", unsafe_allow_html=True
)
