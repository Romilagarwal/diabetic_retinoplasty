import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from swin_transformer import (
    PatchEmbed, WindowAttention, SwinTransformerBlock, 
    MLP, PatchMerging, BasicLayer, SwinTransformer,
    window_partition, window_reverse
)
from tensorflow.keras.saving import register_keras_serializable

# Enable mixed precision before loading model
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

BASE_PATH = r"E:\diabetic-retinoplasty\blindness"
PROCESSED_TEST_DIR = os.path.join(BASE_PATH, "processed", "processed_test_images")
MODEL_PATH = os.path.join(BASE_PATH, "models", "hybrid_model_best.h5")
RESULTS_DIR = os.path.join(BASE_PATH, "results")
TARGET_SIZE = (224, 224)
BATCH_SIZE = 8

os.makedirs(RESULTS_DIR, exist_ok=True)

# Define Cast layer for mixed precision
@register_keras_serializable(package="Custom")
class Cast(tf.keras.layers.Layer):
    """Custom Cast layer to handle mixed precision conversions"""
    def __init__(self, dtype=None, **kwargs):
        super().__init__(**kwargs)
        self.dtype_to_cast = dtype

    def call(self, inputs):
        return tf.cast(inputs, self.dtype_to_cast)

    def get_config(self):
        config = super().get_config()
        config.update({"dtype": self.dtype_to_cast})
        return config

# Update the load_model function
def load_model():
    """Load the trained hybrid model"""
    print(f"Loading model from {MODEL_PATH}")
    
    # Define custom objects dictionary for all custom layers
    custom_objects = {
        'PatchEmbed': PatchEmbed,
        'WindowAttention': WindowAttention,
        'SwinTransformerBlock': SwinTransformerBlock,
        'MLP': MLP,
        'PatchMerging': PatchMerging,
        'BasicLayer': BasicLayer,
        'SwinTransformer': SwinTransformer,
        'window_partition': window_partition,
        'window_reverse': window_reverse,
        'Cast': Cast  # Add Cast layer for mixed precision
    }
    
    # Load model with custom objects scope
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
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

    results_path = os.path.join(RESULTS_DIR, "hybrid_test_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved to {results_path}")

    plt.figure(figsize=(20, 16))
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

    num_samples = min(5, len(filenames))
    sample_indices = np.random.choice(len(filenames), num_samples, replace=False)

    for i, idx in enumerate(sample_indices):
        img_path = os.path.join(PROCESSED_TEST_DIR, filenames[idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        pred_class = predicted_classes[idx]
        pred_probs = predictions[idx]

        plt.subplot(2, num_samples, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Image: {os.path.basename(filenames[idx])}")
        plt.axis('off')

        plt.subplot(2, num_samples, i+1+num_samples)
        bars = plt.bar(range(5), pred_probs)
        bars[pred_class].set_color('red')
        plt.xticks(range(5), class_names, rotation=45)
        plt.title(f"Predicted: {class_names[pred_class]}")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "hybrid_test_predictions_visualization.png"))
    plt.close()

    return results_df

def create_comparison_with_baseline():
    """Compare hybrid model results with baseline EfficientNet results"""
    baseline_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    hybrid_path = os.path.join(RESULTS_DIR, "hybrid_test_predictions.csv")
    
    if os.path.exists(baseline_path) and os.path.exists(hybrid_path):
        baseline_df = pd.read_csv(baseline_path)
        hybrid_df = pd.read_csv(hybrid_path)
        
        comparison_df = pd.merge(
            baseline_df, 
            hybrid_df, 
            on='filename', 
            suffixes=('_baseline', '_hybrid')
        )
        
        baseline_dist = baseline_df['predicted_class'].value_counts().sort_index()
        hybrid_dist = hybrid_df['predicted_class'].value_counts().sort_index()
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        x = np.arange(5)
        width = 0.35
        
        all_classes = list(range(5))
        baseline_counts = [baseline_dist.get(i, 0) for i in all_classes]
        hybrid_counts = [hybrid_dist.get(i, 0) for i in all_classes]
        
        plt.bar(x - width/2, baseline_counts, width, label='EfficientNet')
        plt.bar(x + width/2, hybrid_counts, width, label='Hybrid Model')
        
        plt.xlabel('DR Severity Class')
        plt.ylabel('Count')
        plt.title('Predicted Class Distribution Comparison')
        plt.xticks(x, ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'])
        plt.legend()
        
        plt.subplot(2, 1, 2)
        agreement = (comparison_df['predicted_class_baseline'] == comparison_df['predicted_class_hybrid']).mean() * 100
        disagreement = 100 - agreement
        
        plt.pie(
            [agreement, disagreement], 
            labels=['Agreement', 'Disagreement'], 
            autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336'],
            explode=(0, 0.1)
        )
        plt.title(f'Model Agreement: {agreement:.1f}% of predictions match')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"))
        plt.close()
        
        comparison_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
        print(f"Model comparison saved to {os.path.join(RESULTS_DIR, 'model_comparison.csv')}")
        
        return comparison_df
    else:
        print("Baseline or hybrid results not found. Skipping comparison.")
        return None

def main():
    model = load_model()
    model.summary()

    test_ds, filenames = create_test_dataset()

    results_df = predict_and_visualize(model, test_ds, filenames)

    comparison_df = create_comparison_with_baseline()

    print("Testing complete!")
    print(f"Class distribution in predictions:")
    print(results_df['predicted_class'].value_counts())

    return results_df

if __name__ == "__main__":
    results = main()