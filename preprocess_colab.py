import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from google.colab import drive
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split

drive.mount('/content/drive')

!mkdir -p ~/.kaggle
!cp /content/drive/MyDrive/Kaggle/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

api = KaggleApi()
api.authenticate()

api.competition_download_files('diabetic-retinopathy-detection', path='/content/data')

!unzip /content/data/diabetic-retinopathy-detection.zip -d /content/data

labels_df = pd.read_csv('/content/data/trainLabels.csv')

def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)

    # Extract green channel
    green_channel = image[:, :, 1]

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(green_channel)

    # Denoise using median filter
    denoised_image = cv2.medianBlur(clahe_image, 3)

    # Resize image
    resized_image = cv2.resize(denoised_image, target_size)

    # Normalize pixel values
    normalized_image = resized_image / 255.0

    return normalized_image

def create_dataset(image_paths, labels, batch_size=32):
    def generator():
        for image_path, label in zip(image_paths, labels):
            img = preprocess_image(image_path)
            yield img, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.int32),
        output_shapes=((224, 224), ())
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

image_paths = [f'/content/data/train/{img}.jpeg' for img in labels_df['image']]
labels = labels_df['level'].values

train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

train_dataset = create_dataset(train_paths, train_labels)
val_dataset = create_dataset(val_paths, val_labels)

def unet(input_size=(224, 224, 1)):
    inputs = tf.keras.layers.Input(input_size)

    # Encoder
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bridge
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)

    # Decoder
    up4 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv3)
    up4 = tf.keras.layers.concatenate([up4, conv2])
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up4)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = tf.keras.layers.concatenate([up5, conv1])
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up5)
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv5)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

unet_model = unet()
unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#  Function to apply U-Net segmentation
def apply_unet_segmentation(image):
    prediction = unet_model.predict(np.expand_dims(image, axis=0))
    segmentation_mask = (prediction > 0.5).astype(np.uint8)
    return segmentation_mask[0, :, :, 0]

# Update preprocess_image function to include U-Net segmentation
def preprocess_image_with_unet(image_path, target_size=(224, 224)):
    image = preprocess_image(image_path, target_size)
    segmentation_mask = apply_unet_segmentation(image)
    segmented_image = image * segmentation_mask
    return segmented_image

# Update create_dataset function to use the new preprocessing
def create_dataset_with_unet(image_paths, labels, batch_size=32):
    def generator():
        for image_path, label in zip(image_paths, labels):
            img = preprocess_image_with_unet(image_path)
            yield img, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.int32),
        output_shapes=((224, 224), ())
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset_unet = create_dataset_with_unet(train_paths, train_labels)
val_dataset_unet = create_dataset_with_unet(val_paths, val_labels)

print("Preprocessing complete. Datasets are ready for model training.")
