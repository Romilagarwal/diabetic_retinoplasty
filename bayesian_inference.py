import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from swin_transformer import (
    PatchEmbed, WindowAttention, SwinTransformerBlock, 
    MLP, PatchMerging, BasicLayer, SwinTransformer,
    window_partition, window_reverse
)
from test_hybrid_model import Cast  # Import the Cast layer definition

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Memory growth enabled for GPU")
    except:
        print("Invalid device or cannot modify virtual devices once initialized")

BASE_PATH = r"E:\diabetic-retinoplasty\blindness"
MODEL_PATH = os.path.join(BASE_PATH, "models", "hybrid_model_best.h5")
PROCESSED_TEST_DIR = os.path.join(BASE_PATH, "processed", "processed_test_images")
RESULTS_DIR = os.path.join(BASE_PATH, "results", "bayesian_uncertainty")
os.makedirs(RESULTS_DIR, exist_ok=True)
def enable_dropout_at_inference(model):
    """Enable dropout during inference for Monte Carlo sampling"""
    # This is the updated approach for TF 2.x to enable training mode at inference time
    # which keeps dropout active during prediction
    
    # Create a new model with the same weights but in training mode
    inference_model = tf.keras.models.clone_model(model)
    inference_model.set_weights(model.get_weights())
    
    # Use a custom training-enabled predict function
    def predict_with_dropout(x, **kwargs):
        return inference_model(x, training=True)
        
    return predict_with_dropout

def monte_carlo_predictions(model, img, num_samples=50):
    """Run Monte Carlo Dropout inference for uncertainty estimation"""
    predict_fn = enable_dropout_at_inference(model)
    
    if len(img.shape) == 3:
        img_batch = np.expand_dims(img, axis=0)
    else:
        img_batch = img

    mc_predictions = []

    for _ in tqdm(range(num_samples), desc="Running MC samples", leave=False):
        try:
            # Use the custom predict function that keeps dropout active
            prediction = predict_fn(img_batch)
            
            if isinstance(prediction, list):
                prediction = prediction[0]
                
            mc_predictions.append(prediction[0].numpy() if len(prediction.shape) > 1 else prediction.numpy())
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue

    mc_predictions = np.array(mc_predictions)

    mean_prediction = np.mean(mc_predictions, axis=0)
    uncertainty = np.std(mc_predictions, axis=0)

    return mean_prediction, uncertainty, mc_predictions

def predictive_entropy(mean_prediction):
    """Calculate predictive entropy as uncertainty measure

    Higher entropy = higher uncertainty
    """
    epsilon = 1e-10
    mean_prediction = np.clip(mean_prediction, epsilon, 1.0 - epsilon)

    entropy = -np.sum(mean_prediction * np.log(mean_prediction))
    return entropy

def visualize_uncertainty(image, mean_prediction, uncertainty, class_names=None):
    """Visualize predicted class with uncertainty"""
    if class_names is None:
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

    predicted_class = np.argmax(mean_prediction)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Retinal Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.bar(range(len(mean_prediction)), mean_prediction, yerr=uncertainty,
            capsize=10, color='skyblue', alpha=0.7)
    plt.xticks(range(len(mean_prediction)), class_names, rotation=45)
    plt.title(f'Prediction: {class_names[predicted_class]}\nUncertainty: {uncertainty[predicted_class]:.4f}')
    plt.ylabel('Probability')
    plt.ylim(0, 1.1)

    plt.subplot(1, 3, 3)

    cmap = plt.cm.RdYlGn_r

    confidence = 1.0 - uncertainty
    confidence_normalized = (confidence - min(confidence)) / (max(confidence) - min(confidence) + 1e-10)

    bars = plt.bar(range(len(mean_prediction)), mean_prediction,
                   color=[cmap(conf) for conf in confidence_normalized])

    bars[predicted_class].set_edgecolor('black')
    bars[predicted_class].set_linewidth(2)

    plt.xticks(range(len(mean_prediction)), class_names, rotation=45)
    plt.title('Model Confidence by Class')
    plt.ylabel('Probability')
    plt.ylim(0, 1.1)

    entropy = predictive_entropy(mean_prediction)
    plt.text(0.5, 1.05, f'Predictive Entropy: {entropy:.4f}',
             horizontalalignment='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    return plt.gcf()

def process_test_images(model, test_dir, results_dir, num_samples=50, max_images=None):
    """Process all test images and generate uncertainty metrics"""
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} does not exist!")
        return pd.DataFrame()

    test_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]

    if not test_images:
        print(f"Error: No PNG images found in {test_dir}")
        return pd.DataFrame()

    print(f"Found {len(test_images)} test images")

    if max_images is not None and max_images > 0:
        test_images = test_images[:max_images]
        print(f"Processing {len(test_images)} images (limited by max_images={max_images})")

    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

    results = []

    for img_file in tqdm(test_images, desc="Processing images"):
        img_path = os.path.join(test_dir, img_file)

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error: Could not read image {img_path}")
                continue

            img = cv2.resize(img, (224, 224))
            img_normalized = img / 255.0
            img_normalized = np.expand_dims(img_normalized, axis=-1)
        except Exception as e:
            print(f"Error preprocessing {img_path}: {e}")
            continue

        try:
            mean_pred, uncertainty, mc_samples = monte_carlo_predictions(
                model, img_normalized, num_samples=num_samples)
        except Exception as e:
            print(f"Error during Monte Carlo predictions for {img_path}: {e}")
            continue

        pred_class = np.argmax(mean_pred)
        pred_class_name = class_names[pred_class]
        pred_confidence = mean_pred[pred_class]
        pred_uncertainty = uncertainty[pred_class]

        entropy = predictive_entropy(mean_pred)

        try:
            fig = visualize_uncertainty(img_normalized[:,:,0], mean_pred, uncertainty, class_names)

            output_path = os.path.join(results_dir, f"uncertainty_{os.path.splitext(img_file)[0]}.png")
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)

            if len(results) < 5:
                plt.figure(figsize=(12, 6))
                for i in range(5):
                    sample_idx = np.random.randint(0, len(mc_samples))
                    plt.subplot(1, 5, i+1)
                    plt.bar(range(len(class_names)), mc_samples[sample_idx])
                    plt.title(f"Sample {sample_idx}")
                    plt.xticks(range(len(class_names)), class_names, rotation=90)
                plt.tight_layout()
                mc_path = os.path.join(results_dir, f"mc_samples_{os.path.splitext(img_file)[0]}.png")
                plt.savefig(mc_path)
                plt.close()
        except Exception as e:
            print(f"Error visualizing results for {img_path}: {e}")
            continue

        results.append({
            'image': img_file,
            'predicted_class': pred_class,
            'predicted_class_name': pred_class_name,
            'confidence': pred_confidence,
            'uncertainty': pred_uncertainty,
            'entropy': entropy,
            **{f'prob_class_{i}': mean_pred[i] for i in range(len(class_names))},
            **{f'uncertainty_class_{i}': uncertainty[i] for i in range(len(class_names))},
        })

    if not results:
        print("No results generated! Check error messages above.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, "uncertainty_metrics.csv"), index=False)

    print(f"Results saved to {results_dir}")
    return results_df

def analyze_uncertainty_distribution(results_df, results_dir):
    """Analyze the distribution of uncertainty across predictions"""
    if results_df.empty:
        print("No data to analyze!")
        return

    plt.figure(figsize=(15, 10))

    try:
        plt.subplot(2, 2, 1)
        plt.scatter(results_df['confidence'], results_df['uncertainty'], alpha=0.6)
        plt.xlabel('Confidence')
        plt.ylabel('Uncertainty')
        plt.title('Confidence vs. Uncertainty')
        plt.grid(True, alpha=0.3)

        if len(results_df) > 1:
            z = np.polyfit(results_df['confidence'], results_df['uncertainty'], 1)
            p = np.poly1d(z)
            plt.plot(results_df['confidence'], p(results_df['confidence']), "r--", alpha=0.7)

        plt.subplot(2, 2, 2)
        classes = results_df['predicted_class'].unique()
        for cls in sorted(classes):
            subset = results_df[results_df['predicted_class'] == cls]
            if len(subset) > 0:
                plt.hist(subset['uncertainty'], alpha=0.5, label=f'Class {cls}', bins=min(15, len(subset)))

        plt.xlabel('Uncertainty')
        plt.ylabel('Count')
        plt.title('Uncertainty Distribution by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.hist(results_df['entropy'], bins=min(20, len(results_df)), color='green', alpha=0.7)
        plt.xlabel('Predictive Entropy')
        plt.ylabel('Count')
        plt.title('Distribution of Predictive Entropy')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        data = [results_df[results_df['predicted_class'] == cls]['uncertainty'] for cls in sorted(classes)]
        data = [d for d in data if len(d) > 0]
        if data:
            plt.boxplot(data, labels=[f'Class {cls}' for cls in sorted(classes) if len(results_df[results_df['predicted_class'] == cls]) > 0])
            plt.ylabel('Uncertainty')
            plt.title('Uncertainty by Class (Box Plot)')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "uncertainty_analysis.png"), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error analyzing uncertainty distribution: {e}")

def create_reliability_diagram(results_df, results_dir):
    """Create reliability diagram to assess calibration of uncertainty estimates"""
    if results_df.empty:
        print("No data for reliability diagram!")
        return

    try:
        n_bins = min(10, len(results_df))
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        bin_accuracies = np.zeros(n_bins)
        bin_confidences = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        sample_size = min(50, len(results_df))
        sample_indices = np.random.choice(len(results_df), sample_size, replace=False)

        simulated_true_classes = np.random.randint(0, 5, sample_size)

        for i in sample_indices:
            confidence = results_df.iloc[i]['confidence']
            predicted_class = results_df.iloc[i]['predicted_class']
            true_class = simulated_true_classes[list(sample_indices).index(i)]

            bin_idx = min(int(confidence * n_bins), n_bins - 1)

            bin_confidences[bin_idx] += confidence
            bin_accuracies[bin_idx] += (predicted_class == true_class)
            bin_counts[bin_idx] += 1

        for i in range(n_bins):
            if bin_counts[i] > 0:
                bin_accuracies[i] /= bin_counts[i]
                bin_confidences[i] /= bin_counts[i]

        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

        valid_bins = bin_counts > 0
        plt.plot(bin_confidences[valid_bins], bin_accuracies[valid_bins], 'o-', label='Model')

        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Reliability Diagram (With Simulated Ground Truth)\nNote: Use actual labels for accurate assessment')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "reliability_diagram.png"), bbox_inches='tight')
        plt.close()

        print("Note: The reliability diagram uses simulated ground truth. For accurate calibration assessment, use labeled test data.")
    except Exception as e:
        print(f"Error creating reliability diagram: {e}")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found!")
        return

    if not os.path.exists(PROCESSED_TEST_DIR):
        print(f"Error: Test images directory {PROCESSED_TEST_DIR} not found!")
        return

    try:
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
            
        # The rest of your function remains the same
        results_df = process_test_images(model, PROCESSED_TEST_DIR, RESULTS_DIR, num_samples=50, max_images=None)

        if not results_df.empty:
            analyze_uncertainty_distribution(results_df, RESULTS_DIR)

            create_reliability_diagram(results_df, RESULTS_DIR)

            print("Bayesian uncertainty analysis complete!")
            print(f"Check results in {RESULTS_DIR}")

            avg_uncertainty = results_df['uncertainty'].mean()
            max_uncertainty = results_df['uncertainty'].max()
            min_uncertainty = results_df['uncertainty'].min()

            print(f"\nUncertainty Statistics:")
            print(f"  Average uncertainty: {avg_uncertainty:.4f}")
            print(f"  Maximum uncertainty: {max_uncertainty:.4f}")
            print(f"  Minimum uncertainty: {min_uncertainty:.4f}")

            print("\nClass distribution in predictions:")
            for cls in range(5):
                count = (results_df['predicted_class'] == cls).sum()
                if count > 0:
                    avg_unc = results_df[results_df['predicted_class'] == cls]['uncertainty'].mean()
                    print(f"  Class {cls}: {count} samples, Avg uncertainty: {avg_unc:.4f}")
                else:
                    print(f"  Class {cls}: 0 samples")
        else:
            print("No results generated. Check error messages above.")
    except Exception as e:
        print(f"An error occurred during the analysis: {e}")

if __name__ == "__main__":
    main()
