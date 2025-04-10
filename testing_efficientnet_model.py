import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

BASE_PATH = r"E:\diabetic-retinoplasty\blindness"
PROCESSED_TEST_DIR = os.path.join(BASE_PATH, "processed", "processed_test_images")
MODEL_PATH = os.path.join(BASE_PATH, "models", "efficientnet_best.h5")
RESULTS_DIR = os.path.join(BASE_PATH, "results")
TARGET_SIZE = (224, 224)
BATCH_SIZE = 8

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_model():
    """Load the trained EfficientNetB0 model"""
    print(f"Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def create_test_dataset():
    """Create a TensorFlow dataset from processed test images"""
    print(f"Creating test dataset from {PROCESSED_TEST_DIR}")

    test_images = [f for f in os.listdir(PROCESSED_TEST_DIR) if f.endswith('.png')]
    print(f"Found {len(test_images)} processed test images")

    def test_gen():
        for img_file in test_images:
            img_path = os.path.join(PROCESSED_TEST_DIR, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = img / 255.0
                img = np.expand_dims(img, axis=-1)
                yield img, img_file

    test_dataset = tf.data.Dataset.from_generator(
        test_gen,
        output_signature=(
            tf.TensorSpec(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )
    )

    test_images_ds = test_dataset.map(lambda x, y: x)
    filenames = [item.decode('utf-8') if isinstance(item, bytes) else item for item in test_dataset.map(lambda x, y: y).as_numpy_iterator()]

    test_images_ds = test_images_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return test_images_ds, filenames


def predict_and_visualize(model, test_ds, filenames):
    """Make predictions and visualize results"""
    print("Making predictions on test dataset...")

    predictions = model.predict(test_ds)

    predicted_classes = np.argmax(predictions, axis=1)

    results_df = pd.DataFrame({
        'filename': filenames,
        'predicted_class': predicted_classes
    })

    for i in range(5):
        results_df[f'prob_class_{i}'] = [pred[i] for pred in predictions]

    results_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved to {results_path}")

    plt.figure(figsize=(15, 10))

    num_samples = min(5, len(filenames))
    sample_indices = np.random.choice(len(filenames), num_samples, replace=False)

    for i, idx in enumerate(sample_indices):
        img_path = os.path.join(PROCESSED_TEST_DIR, filenames[idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        pred_class = predicted_classes[idx]
        pred_probs = predictions[idx]

        plt.subplot(2, num_samples, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Image: {filenames[idx]}")
        plt.axis('off')

        plt.subplot(2, num_samples, i+1+num_samples)
        bars = plt.bar(range(5), pred_probs)
        bars[pred_class].set_color('red')
        plt.xticks(range(5), ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'])
        plt.title(f"Predicted: Class {pred_class}")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "test_predictions_visualization.png"))
    plt.close()

    return results_df

def main():
    model = load_model()

    model.summary()

    test_ds, filenames = create_test_dataset()

    results_df = predict_and_visualize(model, test_ds, filenames)

    print("Testing complete!")
    print(f"Class distribution in predictions:")
    print(results_df['predicted_class'].value_counts())

    return results_df


if __name__ == "__main__":
    results = main()
