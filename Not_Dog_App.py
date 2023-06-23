import os
from PIL import Image
import cv2
import numpy as np
from sklearn.svm import OneClassSVM
import streamlit as st

# Set up the model
data_folder = "Training_Data"
subfolders = ["dogs"]
image_rgb = []
X_train = []

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    global X_train
    for subfolder in subfolders:
        subfolder_path = os.path.join(data_folder, subfolder)
        image_paths = get_image_paths(subfolder_path)
        for image_path in image_paths:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (100, 100))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.flatten()
            X_train.append(img)
    X_train = np.array(X_train)
    model = OneClassSVM(nu=0.2)
    model.fit(X_train)
    return model

model = load_model()

# Function to process user input and make predictions
def process_input(image):
    img = cv2.imread(image)
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten()
    prediction = model.predict([img])
    return prediction

# Streamlit app
def main():
    st.title("Image Classification App")
    st.write("Upload an image and the app will make predictions.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        st.write(file_details)

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Make Prediction'):
            image_path = os.path.join('uploads', uploaded_file.name)
            image.save(image_path)
            prediction = process_input(image_path)
            st.write("Prediction:", prediction)

if __name__ == '__main__':
    main()