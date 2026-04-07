# predictor.py
import os
import json
import requests
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder_model.h5")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")

# Put your public Google Drive direct-download links here
MODEL_URL = "https://drive.google.com/uc?export=download&id=1L_yGyl2rar4g6A0gv3e3DjXHSpIQLzJ_"
CONFIG_URL = "https://drive.google.com/uc?export=download&id=1TxsKetjFSTBhe0eohywZJ5eEvYVu7qaz"

def download_file(url, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    session = requests.Session()
    response = session.get(url, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        response = session.get(url, params={"confirm": token}, stream=True)

    response.raise_for_status()

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

def ensure_model_files():
    if not os.path.exists(MODEL_PATH):
        download_file(MODEL_URL, MODEL_PATH)

    if not os.path.exists(CONFIG_PATH):
        download_file(CONFIG_URL, CONFIG_PATH)

@st.cache_resource
def get_model():
    ensure_model_files()
    return load_model(MODEL_PATH)

@st.cache_data
def get_config():
    ensure_model_files()
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def process_input(image_path):
    model = get_model()
    config = get_config()

    image_width = config["image_width"]
    image_height = config["image_height"]
    mse_threshold = config["mse_threshold"]

    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(image_width, image_height)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    decoded_img = model.predict(img_array, verbose=0)
    mse = np.mean(np.square(img_array - decoded_img))

    return 1 if mse < mse_threshold else 0

# import os
# import joblib
# import numpy as np
# from PIL import Image

# def process_input(image_path):
#     target_size = (32, 32)

#     # Load and preprocess the image
#     img = Image.open(image_path).resize(target_size)
#     img_array = np.array(img) / 255.0  # Rescale the image
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
#     # Load the model
#     model = joblib.load("cnn_model.pkl")
    
#     # Predict using the model
#     prediction = model.predict(img_array)
#     return prediction
    
    # # Classify the prediction as 1 (dog) or 0 (not dog)
    # if prediction >= 0.5:
    #     return 1
    # else:
    #     return 0
