import os
from PIL import Image
import streamlit as st
from prediction import process_input
from predictor_model import load_model

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
            # Create the 'uploads' directory if it doesn't exist
            os.makedirs('uploads', exist_ok=True)
            
            image_path = os.path.join('uploads', uploaded_file.name)
            image.save(image_path)
            model = load_model()
            prediction = process_input(image_path, model)
            st.write("Prediction:", prediction)

if __name__ == '__main__':
    main()
