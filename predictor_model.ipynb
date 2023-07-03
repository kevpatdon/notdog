import os
from PIL import Image
import numpy as np
from sklearn.svm import OneClassSVM

data_folder = "Training_Data"
subfolders = ["dogs"]
X_train = []

def load_model():
    global X_train
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
