import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import streamlit as st
import numpy as np


model = load_model('Image_Classification.keras')



data_cat =['apple',
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
 'watermelon']


img_height = 180
img_width = 180


st.title("Image Classification of Vegetables and Fruits")

st.divider()
image = st.text_input("Enter Image: ", 'banana.jpg')


image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat = tf.expand_dims(img_arr, 0)

predict = model.predict(img_bat)


score = tf.nn.softmax(predict)




st.divider()
st.image(image)


st.divider()
st.write("Vegetable/Fruit in image is {} with accuracy of {:.4f}".format(data_cat[np.argmax(score)], np.max(score)*100))
