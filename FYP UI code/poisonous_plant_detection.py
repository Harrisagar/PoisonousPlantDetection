# pylint: disable=E0401
# pylint: disable=no-member

# import os
import cv2
# from tqdm import tqdm
import streamlit as st
# from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the saved model
loaded_model = tf.keras.models.load_model('denoised_images_model.h5')

# Function to preprocess the image
def preprocess_image(img):
    img = img.convert('RGB')  # Ensure image is in RGB format

    # Denoise the image
    img_array = cv2.fastNlMeansDenoisingColored(np.array(img), None, 10, 10, 7, 21)

    # Convert to grayscale and perform histogram equalization
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(gray)

    # Resize the image to match the model input size
    img_array = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR)
    img_array = cv2.resize(img_array, (224, 224))

    # Normalize the image
    img_array = img_array.astype("float32") / 255.0

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make prediction
def predict_image(img_array):
    predictions = loaded_model.predict(img_array)
    return predictions

# Streamlit app
st.title("Poisonous Plant Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(original_image)

    # Resize both images to the same size
    target_size = (original_image.width, original_image.height)
    original_image_resized = original_image.resize(target_size)
    preprocessed_image_resized = Image.fromarray((preprocessed_image[0] * 255).astype(np.uint8))
    preprocessed_image_resized = preprocessed_image_resized.resize(target_size)

    # Display the images side by side
    col1, col2 = st.columns(2)
    col1.image(original_image_resized, caption='Original Image', use_column_width=True)
    col2.image(preprocessed_image_resized, caption='Preprocessed Image', use_column_width=True)

    # Make prediction using preprocessed image
    predictions = predict_image(preprocessed_image)
    predicted_class_index = np.argmax(predictions)

    # Check if predicted_class_index is within a valid range
    if 0 <= predicted_class_index < len(predictions[0]):
        # Assuming the class labels are ordered from 0 to 3
        # If the predicted class index is 2 or 3, consider it as 'Non-Poisonous'
        # Otherwise, consider it as 'Poisonous'
        if predicted_class_index >= 2:
            st.success('Prediction: Non-Poisonous')
        else:
            st.error('Prediction: Poisonous')
    else:
        st.error(f'Error: Unknown class index {predicted_class_index}')
