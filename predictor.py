import os
import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import tensorflow as tf

# Function to classify an image as dog or not dog
image_width = 128
image_height = 128
mse_threshold = 0.057  # Adjust the threshold based on observations

def process_input(image_path):
    model = load_model('autoencoder_model.h5')
    # Load and preprocess the image for prediction
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(image_width, image_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize the image

    # Predict using the loaded autoencoder
    decoded_img = model.predict(img_array)

    # Calculate the mean squared error (MSE) as a measure of reconstruction error
    mse = np.mean(np.square(img_array - decoded_img))

    # Define a threshold to classify as dog or not dog
    if mse < mse_threshold:
        return 1
    else:
        return 0





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
