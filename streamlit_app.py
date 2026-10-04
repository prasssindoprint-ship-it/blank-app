import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set page configuration
st.set_page_config(page_title="ResNet50 Model Tester", layout="centered")

# 1. Load your trained .h5 model
# Cache the model so it doesn't reload on every user interaction
@st.cache_resource
def load_my_model():
    # Replace 'path_to_your_model.h5' with your actual file name
    model = tf.keras.models.load_model('path_to_your_model.h5')
    return model

try:
    model = load_my_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please ensure your .h5 file is in the same directory or provide the correct path.")

# 2. UI Elements
st.title("🖼️ ResNet50 Model Testing Dashboard")
st.write("Upload an image to see your model's prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# 3. Image Preprocessing and Prediction
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("⚙️ Processing and predicting...")
    
    # ResNet50 default input size is 224x224
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convert image to numpy array
    img_array = np.asarray(image)
    
    # Ensure image has 3 channels (RGB)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
        
    # Add batch dimension (1, 224, 224, 3)
    img_reshape = np.expand_dims(img_array, axis=0)
    
    # Apply ResNet50 specific preprocessing (scales pixels appropriately)
    # Use this if your model was trained using tf.keras.applications.resnet50.preprocess_input
    preprocess_input = tf.keras.applications.resnet50.preprocess_input
    prepared_image = preprocess_input(img_reshape)
    
    # Make prediction
    predictions = model.predict(prepared_image)
    
    # 4. Display Results
    # Custom Logic: If your model is custom-trained, map predictions to your classes
    st.subheader("Results")
    
    # Example for binary/multi-class outputs (Change this based on your specific model output)
    st.write("Raw Model Output probabilities:", predictions)
    
    # If using standard ImageNet classes, uncomment the lines below:
    # decode_predictions = tf.keras.applications.resnet50.decode_predictions
    # label = decode_predictions(predictions, top=3)[0]
    # for idx, res in enumerate(label):
    #     st.write(f"**{idx+1}. {res[1]}**: {res[2]*100:.2f}%")
