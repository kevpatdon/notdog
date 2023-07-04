import os
from PIL import Image
import streamlit as st
from prediction import process_input
from predictor_model import load_model

def main():
    st.title("DOG? NOT DOG?")
    st.write("Upload an image and the app will determine whether this is an image of a dog.")

    uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files is not None:
        dog_count = 4000
        not_dog_count = 4000

        model = load_model()  # Load the model

        os.makedirs("uploads", exist_ok=True)  # Create the "uploads" directory

        for uploaded_file in uploaded_files:
            dog_count += 1
            not_dog_count += 1

            if dog_count > 4010 or not_dog_count > 4010:
                break

            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}

            image_path = os.path.join("uploads", uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            image = Image.open(image_path)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            if st.button('Is that a dog?'):
                prediction = process_input(image_path, model)

                if prediction == "dogs":
                    st.image('dog.jpg', caption='That is a dog.')

                    correct = st.button('Correct')
                    if not correct:
                        dogs_folder = "Training_Data/dogs"
                        new_filename = f"dog.{dog_count}"
                        new_image_path = os.path.join(dogs_folder, new_filename + ".jpg")
                        image.save(new_image_path)
                else:
                    st.image('notdog.jpg', caption='That is not a dog.')

                    correct = st.button('Correct')
                    if not correct:
                        not_dogs_folder = "Training_Data/not_dogs"
                        new_filename = f"cat.{not_dog_count}"
                        new_image_path = os.path.join(not_dogs_folder, new_filename + ".jpg")
                        image.save(new_image_path)

            # Remove the uploaded file after processing
            os.remove(image_path)



if __name__ == '__main__':
    main()
