# autoencoder model for anomaly detection
import os
import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

image_width = 128
image_height = 128
model_filename = 'autoencoder_model.h5'

# Path to your training data
train_data_dir = 'Training_Data/dogs'

# Data generator for training
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_width, image_height),
    batch_size=100,
    class_mode='input',  # Use 'input' for autoencoder
    shuffle=True
)

# Define the autoencoder model
input_img = Input(shape=(image_width, image_height, 3))
x = Flatten()(input_img)
encoded = Dense(128, activation='relu')(x)
decoded = Dense(image_width * image_height * 3, activation='sigmoid')(encoded)
decoded = Reshape((image_width, image_height, 3))(decoded)

autoencoder = Model(input_img, decoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(train_generator, epochs = 20)

# Save the model to a HDF5 file
autoencoder.save(model_filename)

# Test
mse_threshold = 0.07  # Adjust the threshold based on observations

# Function to classify an image as dog or not dog
def classify_image(image_path, model):
    # Load and preprocess the image for prediction
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(image_width, image_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize the image

    # Predict using the loaded autoencoder
    decoded_img = model.predict(img_array)

    # Calculate the mean squared error (MSE) as a measure of reconstruction error
    mse = np.mean(np.square(img_array - decoded_img))

    # Define a threshold to classify as dog or not dog
    if mse < mse_threshold:
        return "dog", mse
    else:
        return "not dog", mse

# Directory containing test images
test_data_dir = 'Training_Data/test_dogs/dog'

# Ensure the directory exists
assert os.path.exists(test_data_dir), f"Directory '{test_data_dir}' not found."

# Initialize counters
dog_count = 0
not_dog_count = 0

# List all files in the directory
test_image_paths = [os.path.join(test_data_dir, fname) for fname in os.listdir(test_data_dir)]

# Classify each image in the directory
for image_path in test_image_paths:
    classification, mse = classify_image(image_path, autoencoder)
    if classification == "dog":
        dog_count += 1
    elif classification == "not dog":
        not_dog_count += 1
    print(f"{classification.capitalize()} - MSE: {mse:.4f} - {image_path}")

# Print the final counts
print(f"\nTotal Dogs: {dog_count}")
print(f"Total Not Dogs: {not_dog_count}")






























# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# import joblib

# data_folder = "Training_Data"
# subfolders = ["dogs", "test_dogs"]
# target_size = (32, 32)
# batch_size = 32
# epochs = 10

# # Create ImageDataGenerator for data augmentation and normalization
# data_generator = ImageDataGenerator(rescale=1.0 / 255.0)

# # Load and prepare the training data
# train_data = data_generator.flow_from_directory(
#     data_folder,
#     classes=[subfolders[0]],
#     target_size=target_size,
#     batch_size=batch_size,
#     class_mode="binary"
# )

# # Load and prepare the testing data
# test_data = data_generator.flow_from_directory(
#     data_folder,
#     classes=[subfolders[1]],
#     target_size=target_size,
#     batch_size=batch_size,
#     class_mode="binary"
# )

# # Build the model
# model = tf.keras.Sequential([
#     Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu', padding='same'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(train_data, epochs=epochs)

# # Evaluate the model
# loss, accuracy = model.evaluate(test_data)

# # Make predictions
# predictions = model.predict(test_data)

# # Export model
# joblib.dump(model, 'cnn_model.pkl')
