import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

def process_input(image_path, model):
    target_size = (32, 32)
    
    # Load and preprocess the image
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)
    
    # Perform prediction using the model
    prediction = model.predict(img_array)
    prediction = prediction[0][0]
    
    # Classify the prediction as 1 (dog) or 0 (not dog)
    if prediction >= 0.5:
        return 1
    else:
        return 0
