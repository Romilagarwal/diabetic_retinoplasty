import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_PATH = r"E:\diabetic-retinoplasty\blindness"
CSV_PATH = os.path.join(BASE_PATH, "train.csv")
IMAGE_DIR = os.path.join(BASE_PATH, "train_images")
OUTPUT_DIR = os.path.join(BASE_PATH, "processed")
PROCESSED_IMAGES_DIR = os.path.join(OUTPUT_DIR, "processed_images")
TARGET_SIZE = (224, 224)
BATCH_SIZE = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)

def ben_graham_preprocessing(image, sigmaX=10):
    """Apply Ben Graham's preprocessing technique"""
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image

def preprocess_image(image_path, save_path=None):
    """Preprocessing pipeline for retinal images"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None

        green_channel = image[:, :, 1]

        enhanced = ben_graham_preprocessing(green_channel)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)

        denoised = cv2.medianBlur(enhanced, 5)

        resized = cv2.resize(denoised, TARGET_SIZE)

        if save_path:
            cv2.imwrite(save_path, resized)

        normalized = resized / 255.0

        return normalized[..., np.newaxis]

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_all_images(df):
    """Process all images and save the results"""
    print(f"Processing {len(df)} images...")

    for diagnosis in range(5):
        os.makedirs(os.path.join(PROCESSED_IMAGES_DIR, str(diagnosis)), exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['id_code']
        diagnosis = row['diagnosis']

        img_path = os.path.join(IMAGE_DIR, f"{img_id}.png")
        save_path = os.path.join(PROCESSED_IMAGES_DIR, str(diagnosis), f"{img_id}_processed.png")

        preprocess_image(img_path, save_path)

    print(f"All images processed and saved to {PROCESSED_IMAGES_DIR}")

def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples from {CSV_PATH}")
    print(f"Class distribution:\n{df['diagnosis'].value_counts()}")

    train_split, val_split = train_test_split(
        df, test_size=0.2, stratify=df['diagnosis'], random_state=42
    )

    train_df = pd.DataFrame(train_split)
    val_df = pd.DataFrame(val_split)

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val_split.csv"), index=False)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    process_all_images(df)

    def train_gen():
        for _, row in train_df.iterrows():
            img_id = row['id_code']
            img_path = os.path.join(IMAGE_DIR, f"{img_id}.png")
            label = row['diagnosis']
            processed = preprocess_image(img_path)
            if processed is not None:
                yield processed, label

    train_dataset = tf.data.Dataset.from_generator(
        train_gen,
        output_signature=(
            tf.TensorSpec(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def val_gen():
        for _, row in val_df.iterrows():
            img_id = row['id_code']
            img_path = os.path.join(IMAGE_DIR, f"{img_id}.png")
            label = row['diagnosis']
            processed = preprocess_image(img_path)
            if processed is not None:
                yield processed, label

    val_dataset = tf.data.Dataset.from_generator(
        val_gen,
        output_signature=(
            tf.TensorSpec(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    sample_df = df.sample(min(5, len(df)))
    plt.figure(figsize=(15, 6))

    for i, (_, row) in enumerate(sample_df.iterrows()):
        img_path = os.path.join(IMAGE_DIR, f"{row['id_code']}.png")
        original = cv2.imread(img_path)
        processed = preprocess_image(img_path)

        if original is not None and processed is not None:
            plt.subplot(2, 5, i+1)
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.title(f"Original\nClass: {row['diagnosis']}")

            plt.subplot(2, 5, i+6)
            plt.imshow(processed[:, :, 0], cmap='gray')
            plt.title("Processed")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "preprocessing_samples.png"))
    plt.close()

    print(f"Preprocessing complete.")

    return train_dataset, val_dataset


if __name__ == "__main__":
    train_ds, val_ds = main()
