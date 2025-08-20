import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Load the model
model = load_model('Image_Classification.keras')

# Class labels
data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon'
]

# Image size expected by the model
img_height = 180
img_width = 180

# Streamlit App
st.title("🥦🍎 Image Classification of Vegetables and Fruits")
st.divider()

# Let user choose input method
input_method = st.radio("Choose input method", ["Upload Image", "Paste Image URL"])

image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif input_method == "Paste Image URL":
    image_url = st.text_input("Paste image URL here")
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error(f"Unable to fetch or open the image: {e}")

if image is not None:
    try:
        # Resize and preprocess
        image = image.resize((img_width, img_height))
        img_array = tf.keras.utils.img_to_array(image)
        img_batch = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_batch)
        score = tf.nn.softmax(prediction[0])

        # Display results
        st.image(image, caption="Input Image", use_column_width=True)
        st.divider()
        st.success(f"🧠 Prediction: **{data_cat[np.argmax(score)]}**")
        st.write(f"✅ Confidence: **{np.max(score) * 100:.2f}%**")

    except Exception as e:
        st.error(f"Error processing the image: {e}")
