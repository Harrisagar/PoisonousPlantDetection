from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Load your Keras models
model1 = tf.keras.models.load_model("Final_models/20epoch_backgroundremoved_model.h5")
model2 = tf.keras.models.load_model("Final_models/20epoch_denoised_model_with_more_layers.h5")
model3 = tf.keras.models.load_model("Final_models/20epoch_equalized_model.h5")

# Define the list of classes
li = ['Abrus precatorius', 'Caladium Bicolor', 'Curcas', 'Datura Stramonium', 
      'Gloriosa Superba', 'Jatropha Multifida', 'Not_Poisonous', 
      'Ricinus Communis', 'Thevetia Peruviana']

# Function to preprocess an image
def preprocess_image(image):
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize the pixel values
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict a single image using ensemble method
def predict_single_image(image, models, class_labels):
    # Preprocess the image
    img_array = preprocess_image(image)

    # Initialize dictionary to count votes for each class
    class_counts = {class_label: 0 for class_label in class_labels}

    # Iterate through predictions of each model
    for model in models:
        prediction = model.predict(img_array)
        max_prob_index = np.argmax(prediction)
        predicted_class = class_labels[max_prob_index]

        # Increment count for the predicted class
        class_counts[predicted_class] += 1

    # Select the class with the highest count as the final predicted class
    final_predicted_class = max(class_counts, key=class_counts.get)
    
    return final_predicted_class

# Streamlit app
st.title("Poisonous Plant Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(original_image)
    
    # Resize the preprocessed image to match the width of the original image
    width = original_image.width
    preprocessed_image_resized = Image.fromarray((preprocessed_image[0] * 255).astype(np.uint8)).resize((width, width), Image.ANTIALIAS)
    
    # Display the original and preprocessed images side by side
    col1, col2 = st.columns(2)
    col1.image(original_image, caption='Original Image', use_column_width=True)
    col2.image(preprocessed_image_resized, caption='Preprocessed Image', use_column_width=True)

    # Predict the class of the single image
    predicted_class = predict_single_image(original_image, [model1, model2, model3], li)

    # Show the predicted class
    if predicted_class in ['Abrus precatorius', 'Caladium Bicolor', 'Curcas', 'Datura Stramonium', 'Gloriosa Superba', 'Jatropha Multifida']:
        st.error(f'Prediction: Poisonous ({predicted_class})')
    else:
        st.success(f'Prediction: Non-Poisonous ({predicted_class})')
