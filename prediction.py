import os
from PIL import Image
import numpy as np
from sklearn.svm import OneClassSVM

def process_input(image, model):
    img = Image.open(image)
    img = img.resize((100, 100))
    img = img.convert("L")
    img = np.array(img)
    img = img.flatten()
    prediction = model.predict([img])
    return prediction
