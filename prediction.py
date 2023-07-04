import os
from PIL import Image
import numpy as np

def process_input(image_path, model):
    try:
        img = Image.open(image_path)
        img = img.resize((100, 100))
        img = img.convert("L")
        img = np.array(img)
        img = img.flatten()
        prediction = model.predict([img])
        return prediction
    except (OSError, IOError):
        raise ValueError("Invalid image file or unable to open the image.")
