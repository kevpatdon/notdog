**Overview**

This project involves developing an image classifier application that can determine whether an uploaded image is of a dog or not. The classifier leverages an autoencoder trained exclusively on images of dogs. The primary objective is to identify images that deviate significantly from the dog class, which the autoencoder was trained on, indicating that the image is likely not a dog.


**Key Features**

•	Image Upload Interface: A user-friendly interface for uploading images for classification.
•	Autoencoder-Based Classification: Utilizes an autoencoder neural network to reconstruct input images and measure reconstruction error to classify images.
•	Dog vs. Not-Dog Detection: Classifies images based on reconstruction error, where images with high reconstruction error (indicating poor reconstruction) are classified as "Not-Dog."


**Technical Details**

•	Autoencoder Architecture: The autoencoder consists of an encoder and decoder, where the encoder compresses the image into a lower-dimensional latent space and the decoder reconstructs the image from this latent representation.
•	Training Dataset: The autoencoder is trained on a dataset comprising only images of dogs, ensuring the model learns to effectively reconstruct dog images.
•	Reconstruction Error Analysis: After training, the reconstruction error for dog images (low error) and non-dog images (high error) is used to set a threshold for classification.


**Tools and Technologies**

•	Python: Programming language for developing the model and application.
•	TensorFlow/Keras: Frameworks for building and training the autoencoder model.
•	Streamlit: Framework for developing the web-based image upload and classification interface.
•	Pillow (PIL): Library for image preprocessing and manipulation.
•	NumPy: Library for numerical computations and handling image arrays.
•	OS: Library for file path manipulations and directory handling.


**Possible Use Cases**

•	Pet Adoption Platforms: Automatically filter and identify dog images from user uploads.
•	Social Media Apps: Tag and categorize images accurately as dog or non-dog.
•	Surveillance Systems: Detect the presence of dogs in surveillance footage for security or monitoring purposes.


This project demonstrates the application of autoencoders in image classification tasks, leveraging unsupervised learning to distinguish images based on their reconstruction fidelity.
