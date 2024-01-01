import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model('path/to/your/saved/my_model.h5')

def predict(image):
    # Preprocess the image (ensure this matches the training preprocessing)
    # Resize, scale, etc.
    image = np.array(image)  # Convert image to numpy array
    # Perform necessary preprocessing steps

    # Make prediction
    prediction = model.predict(image)

    # Return prediction or whatever information you need
    return prediction

# Set up your Streamlit application
def main():
    st.title('Mammal Image Classification')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")

        # On a button click, make prediction
        if st.button('Predict'):
            # Display the prediction
            prediction = predict(image)
            st.write('The predicted class is:', prediction)

if __name__ == '__main__':
    main()
