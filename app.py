import streamlit as st
from tensorflow.python.keras.models import load_model
import tensorflow
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import cv2
from matplotlib.image import imread
from PIL import Image

from tensorflow.python.keras import backend
backend.clear_session()
model = load_model('potato_detector.h5')

def load_image(img_file):
	image = Image.open(img_file).resize((128,128))
	image = np.array(image)
	return np.expand_dims(image, axis=0)

def run_model(img_file):
	original_image = Image.open(img_file)
	image = np.array(original_image.resize((128,128)))
	image = np.expand_dims(image, axis=0)
	

	result = model.predict(image).flatten()[0]

	st.header('Here is the model prediction:')
	if result == 1:
		st.write('We got ourselves a Potato over here!!!!')
	else:
		st.write('No potatoes here!!!!')

	#plt.imshow(imread(img_file))
	width, height = original_image.size
	if width > 512 or height > 512:
		st.image(img_file,width = 512)
	else:
		st.image(img_file)

st.title('Potato Not Potato')
st.header('Created by Stephanie "Potato" Rodriguez')
st.subheader('Welcome to my sophisticated, state-of-the-art convolutional neural network.')
st.subheader('With high accuracy, it will predict if something is a potato or not')



st.header('Test it out yourself!')
img_file_buffer = st.file_uploader("Upload your own image", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    run_model(img_file_buffer)
    img_file_buffer = None
