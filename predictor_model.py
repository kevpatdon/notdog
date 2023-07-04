import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_model():
    data_folder = "Training_Data"
    subfolders = ["dogs", "test_dogs"]
    target_size = (32, 32)
    batch_size = 32
    epochs = 10

    # Create ImageDataGenerator for data augmentation and normalization
    data_generator = ImageDataGenerator(rescale=1.0 / 255.0)

    # Load and prepare the training data
    train_data = data_generator.flow_from_directory(
        os.path.join(data_folder, subfolders[0]),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary"
    )

    # Build the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 
