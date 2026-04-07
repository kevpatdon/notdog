# app.py
import os
from PIL import Image, UnidentifiedImageError
import streamlit as st
from predictor import process_input

def main():
    st.set_page_config(layout="centered")

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffffff;
            padding: 1rem;
            color: #555555;
        }
        .stApp h1 {
            color: #555555;
        }
        .stApp .stButton>button {
            color: #ffffff;
        }
        .stImage img {
            object-fit: contain;
        }
        .stApp .stMarkdown {
            color: #555555;
        }
        .stApp .stAlert {
            color: #555555;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("DOG? NOT DOG?")
    st.write("Upload an image and the app will determine whether this is an image of a dog.")

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

            if st.button("Is that a dog?"):
                os.makedirs("uploads", exist_ok=True)
                image_path = os.path.join("uploads", uploaded_file.name)
                image.save(image_path)

                prediction = process_input(image_path)

                if prediction == 1:
                    st.image("dog.jpg")
                    st.write("That is a dog.")
                else:
                    st.image("notdog.jpg")
                    st.write("That is not a dog.")

        except UnidentifiedImageError:
            st.error("The uploaded file is not a valid image. Please upload a proper image file.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
