import os
from PIL import Image
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

def get_image_paths(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_paths.append(os.path.join(folder_path, filename))
    return image_paths

data_folder = "Training_Data"
subfolders = ["dogs", "not_dogs"]

def load_model():
    X_train = []  # Reset X_train list for each model loading
    y_train = []
    for label, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(data_folder, subfolder)
        image_paths = get_image_paths(subfolder_path)
        for image_path in image_paths:
            img = Image.open(image_path)
            img = img.resize((100, 100))
            img = img.convert("L")
            img = np.array(img)
            img = img.flatten()
            X_train.append(img)
            y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.3,
        "max_depth": 3
    }

    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, "Validation")])

    return model
