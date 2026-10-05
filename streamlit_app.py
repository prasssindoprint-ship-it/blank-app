import urllib.request
import tempfile
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# 1. App Configuration & Title
st.set_page_config(page_title="ResNet50 Model Tester", layout="centered")
st.title("🖼️ ResNet50 .h5 Model Tester")
st.write("Load a custom ResNet50 `.h5` model from a URL and test it with images.")

# 2. Sidebar Inputs for Model URL
st.sidebar.header("Model Configuration")
model_url = st.sidebar.text_input(
    "Enter ResNet50 .h5 Model URL:",
    value="https://googleapis.com"
)

# Cache the model loading so it doesn't redownload on every interaction
@st.cache_resource
def load_model_from_url(url):
    try:
        with st.spinner("Downloading and loading model... This may take a moment."):
            # Create a temporary file to save the downloaded .h5 file
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
                urllib.request.urlretrieve(url, tmp_file.name)
                # Load the model structure + weights
                # Note: If your file only contains weights, use tf.keras.applications.ResNet50() and load_weights() instead
                model = tf.keras.models.load_model(tmp_file.name)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model if URL is provided
if model_url:
    model = load_model_from_url(model_url)
else:
    model = None
    st.info("Please enter a valid model URL in the sidebar.")

# 3. Image Upload and Prediction Pipeline
if model is not None:
    uploaded_file = st.file_uploader("Choose an image to test...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess the image for ResNet50 (Target size: 224x224)
        img_resized = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)
        
        # Run prediction
        if st.button("Predict Class"):
            with st.spinner("Running inference..."):
                preds = model.predict(img_preprocessed)
                
                # Try decoding predictions (Works out-of-the-box if using ImageNet labels)
                try:
                    decoded_preds = decode_predictions(preds, top=3)[0]
                    st.subheader("Top Predictions:")
                    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
                        st.write(f"**{i+1}. {label.replace('_', ' ').title()}**: {score*100:.2f}%")
                except Exception:
                    # Fallback for custom trained models with custom output shapes
                    st.subheader("Raw Output (Custom Classes):")
                    predicted_class = np.argmax(preds, axis=1)[0]
                    st.write(f"Predicted Class Index: **{predicted_class}**")
                    st.write("Raw probabilities:", preds)
