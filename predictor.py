# predictor.py
import os
import json
import hashlib
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder_model.h5")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")

MODEL_GDRIVE_ID = "1L_yGyl2rar4g6A0gv3e3DjXHSpIQLzJ_"
CONFIG_GDRIVE_ID = "1TxsKetjFSTBhe0eohywZJ5eEvYVu7qaz"

EXPECTED_MODEL_MD5 = "7d540393af94c38f19b8c1f90fd7b752"
EXPECTED_CONFIG_MD5 = "cc3bf3c1be792bf8ae88c02fd3ee5eba"


def is_valid_hdf5(file_path):
    if not os.path.exists(file_path):
        return False

    with open(file_path, "rb") as f:
        signature = f.read(8)

    return signature == b"\x89HDF\r\n\x1a\n"


def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def is_valid_md5(file_path, expected_md5):
    if not os.path.exists(file_path):
        return False
    return get_md5(file_path) == expected_md5


def download_with_gdown(file_id, output_path):
    try:
        import gdown
    except ImportError as e:
        raise ImportError(
            "gdown is not installed. Add gdown to requirements.txt."
        ) from e

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url=url, output=output_path, quiet=False, fuzzy=True)


def ensure_config_file():
    if not os.path.exists(CONFIG_PATH):
        with st.spinner("Downloading config file..."):
            download_with_gdown(CONFIG_GDRIVE_ID, CONFIG_PATH)

    if not is_valid_md5(CONFIG_PATH, EXPECTED_CONFIG_MD5):
        if os.path.exists(CONFIG_PATH):
            os.remove(CONFIG_PATH)

        with st.spinner("Redownloading config file..."):
            download_with_gdown(CONFIG_GDRIVE_ID, CONFIG_PATH)

        if not is_valid_md5(CONFIG_PATH, EXPECTED_CONFIG_MD5):
            raise ValueError(
                "Downloaded config file failed MD5 validation. "
                "Check your Google Drive sharing settings or file ID."
            )


def ensure_model_file():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model file..."):
            download_with_gdown(MODEL_GDRIVE_ID, MODEL_PATH)

    if not is_valid_hdf5(MODEL_PATH):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        with st.spinner("Redownloading model file..."):
            download_with_gdown(MODEL_GDRIVE_ID, MODEL_PATH)

        if not is_valid_hdf5(MODEL_PATH):
            raise ValueError(
                "Downloaded model file is not a valid HDF5 file. "
                "Google Drive may be returning the wrong content."
            )

    if not is_valid_md5(MODEL_PATH, EXPECTED_MODEL_MD5):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        with st.spinner("Redownloading model file after checksum mismatch..."):
            download_with_gdown(MODEL_GDRIVE_ID, MODEL_PATH)

        if not is_valid_md5(MODEL_PATH, EXPECTED_MODEL_MD5):
            raise ValueError(
                "Downloaded model file failed MD5 validation. "
                "Check your Google Drive sharing settings or file ID."
            )


def ensure_model_files():
    os.makedirs(MODEL_DIR, exist_ok=True)
    ensure_config_file()
    ensure_model_file()


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
        image_path,
        target_size=(image_width, image_height)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    decoded_img = model.predict(img_array, verbose=0)
    mse = np.mean(np.square(img_array - decoded_img))

    if mse < mse_threshold:
        return 1
    else:
        return 0


# local run
# import os
# import json
# import numpy as np
# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "autoencoder_model.h5")
# CONFIG_PATH = os.path.join(BASE_DIR, "model_config.json")

# @st.cache_resource
# def get_model():
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
#     return load_model(MODEL_PATH)

# @st.cache_data
# def get_config():
#     if not os.path.exists(CONFIG_PATH):
#         raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
#     with open(CONFIG_PATH, "r") as f:
#         return json.load(f)

# def process_input(image_path):
#     model = get_model()
#     config = get_config()

#     image_width = config["image_width"]
#     image_height = config["image_height"]
#     mse_threshold = config["mse_threshold"]

#     img = tf.keras.preprocessing.image.load_img(
#         image_path,
#         target_size=(image_width, image_height)
#     )
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0

#     decoded_img = model.predict(img_array, verbose=0)
#     mse = np.mean(np.square(img_array - decoded_img))

#     return 1 if mse < mse_threshold else 0
