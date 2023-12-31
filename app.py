import os
from PIL import Image
import streamlit as st
from predictor import process_input
from predictor_model import train_model

def main():
    st.title("DOG? NOT DOG?")
    st.write("Upload an image and the app will determine whether this is an image of a dog.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        # st.write(file_details)

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Is that a dog?'):
            # Create the 'uploads' directory if it doesn't exist
            os.makedirs('uploads', exist_ok=True)

            image_path = os.path.join('uploads', uploaded_file.name)
            image.save(image_path)

            model = train_model()  # Load the model
            prediction = process_input(image_path, model)

            if prediction == 1:
                st.image('dog.jpg', caption='That is a dog.')
            else:
                st.image('notdog.jpg', caption='That is not a dog.')

if __name__ == '__main__':
    main()
