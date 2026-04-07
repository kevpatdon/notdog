# predictor_model.py
import os
import json
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
MODEL_FILENAME = "autoencoder_model.h5"
THRESHOLD_FILENAME = "model_config.json"
TRAIN_DATA_DIR = "Data/train_dogs"

def build_autoencoder():
    input_img = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    x = Flatten()(input_img)
    encoded = Dense(128, activation="relu")(x)
    decoded = Dense(IMAGE_WIDTH * IMAGE_HEIGHT * 3, activation="sigmoid")(encoded)
    decoded = Reshape((IMAGE_WIDTH, IMAGE_HEIGHT, 3))(decoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    return autoencoder

def train_and_save():
    if not os.path.exists(TRAIN_DATA_DIR):
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DATA_DIR}")

    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=32,
        class_mode="input",
        shuffle=True
    )

    if train_generator.samples == 0:
        raise ValueError(
            f"No training images found in {TRAIN_DATA_DIR}. "
            f"Expected structure like Data/train_dogs/dogs/*.jpg"
        )

    model = build_autoencoder()
    model.fit(train_generator, epochs=20)

    model.save(MODEL_FILENAME)

    config = {
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "mse_threshold": 0.068,
        "model_filename": MODEL_FILENAME
    }

    with open(THRESHOLD_FILENAME, "w") as f:
        json.dump(config, f)

    print(f"Saved model to {MODEL_FILENAME}")
    print(f"Saved config to {THRESHOLD_FILENAME}")

if __name__ == "__main__":
    train_and_save()

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
