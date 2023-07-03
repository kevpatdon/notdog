import os
from PIL import Image
import numpy as np
from sklearn.svm import OneClassSVM

def get_image_paths(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_paths.append(os.path.join(folder_path, filename))
    return image_paths

data_folder = "Training_Data"
subfolders = ["dogs"]

def load_model():
    X_train = []  # Reset X_train list for each model loading
    for subfolder in subfolders:
        subfolder_path = os.path.join(data_folder, subfolder)
        image_paths = get_image_paths(subfolder_path)
        for image_path in image_paths:
            img = Image.open(image_path)
            img = img.resize((100, 100))
            img = img.convert("L")
            img = np.array(img)
            img = img.flatten()
            X_train.append(img)
    X_train = np.array(X_train)
    model = OneClassSVM(nu=0.2)
    model.fit(X_train)
    return model
