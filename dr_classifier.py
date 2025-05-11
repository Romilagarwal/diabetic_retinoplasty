import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2
import glob

# Use the same constants as in your GAN code
BASE_PATH = r"E:\diabetic-retinoplasty\blindness"
PROCESSED_IMAGES_DIR = os.path.join(BASE_PATH, "processed", "processed_images")
GAN_IMAGES_DIR = os.path.join(BASE_PATH, "processed", "gan_synthetic_images")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "models")
RESULTS_PATH = os.path.join(BASE_PATH, "results")
IMAGE_SIZE = (224, 224)
CHANNELS = 1  
NUM_CLASSES = 5
BATCH_SIZE = 16  # Reduced from 32
EPOCHS = 30

os.makedirs(RESULTS_PATH, exist_ok=True)

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Memory growth enabled for GPU")
    except:
        print("Failed to enable memory growth")

def build_combined_dataset(use_synthetic=True, synthetic_ratio=1.0):
    """
    Build a combined dataset from original and synthetic images.
    
    Args:
        use_synthetic: Whether to include synthetic images
        synthetic_ratio: Ratio of synthetic images to include (1.0 means all)
    
    Returns:
        train_ds, val_ds, test_ds: TensorFlow datasets for training, validation and testing
    """
    all_images = []
    all_labels = []
    source_types = []  # 0 for original, 1 for synthetic
    
    # Load original images
    print("Loading original images:")
    for class_idx in range(NUM_CLASSES):
        class_dir = os.path.join(PROCESSED_IMAGES_DIR, str(class_idx))
        if not os.path.exists(class_dir):
            print(f"Warning: Original directory for class {class_idx} does not exist!")
            continue
        
        image_files = glob.glob(os.path.join(class_dir, "*.png"))
        print(f"Class {class_idx}: Found {len(image_files)} original images")
        
        for img_path in image_files:
            all_images.append(img_path)
            all_labels.append(class_idx)
            source_types.append(0)  # Original image
    
    # Load synthetic images if enabled
    if use_synthetic:
        print("\nLoading synthetic images:")
        for class_idx in range(NUM_CLASSES):
            class_dir = os.path.join(GAN_IMAGES_DIR, str(class_idx))
            if not os.path.exists(class_dir):
                print(f"Warning: Synthetic directory for class {class_idx} does not exist!")
                continue
            
            image_files = glob.glob(os.path.join(class_dir, "*.png"))
            
            # Apply synthetic ratio if less than 1.0
            if synthetic_ratio < 1.0:
                synthetic_count = int(len(image_files) * synthetic_ratio)
                np.random.shuffle(image_files)
                image_files = image_files[:synthetic_count]
            
            print(f"Class {class_idx}: Using {len(image_files)} synthetic images")
            
            for img_path in image_files:
                all_images.append(img_path)
                all_labels.append(class_idx)
                source_types.append(1)  # Synthetic image
    
    # Split the dataset into training, validation, and test sets
    train_images, temp_images, train_labels, temp_labels, train_sources, temp_sources = train_test_split(
        all_images, all_labels, source_types, test_size=0.3, stratify=all_labels, random_state=42
    )
    
    val_images, test_images, val_labels, test_labels, val_sources, test_sources = train_test_split(
        temp_images, temp_labels, temp_sources, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    # Print dataset statistics
    print("\nDataset statistics:")
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    print(f"Test set: {len(test_images)} images")
    
    # Print class distribution
    class_distribution = {}
    for label in train_labels:
        class_distribution[label] = class_distribution.get(label, 0) + 1
    
    print("\nTraining set class distribution:")
    for class_idx in range(NUM_CLASSES):
        count = class_distribution.get(class_idx, 0)
        print(f"Class {class_idx}: {count} images")
    
    # Create TensorFlow datasets
    def preprocess_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=CHANNELS)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        
        # If using grayscale, convert to RGB for transfer learning models
        if CHANNELS == 1:
            img = tf.tile(img, [1, 1, 3])
        
        return img
    
    def load_and_preprocess(img_path, label, source):
        return preprocess_image(img_path), label, source
    
    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_sources))
    train_ds = train_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y, z: (x, y))  # Discard source info for training
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels, val_sources))
    val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y, z: (x, y))  # Discard source info for validation
    
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_sources))
    test_ds = test_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y, z: (x, y))  # Discard source info for testing
    
    # Configure datasets for performance - add drop_remainder=True
    train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE) 
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

def build_classifier_model():
    """Build and compile a classifier model using EfficientNetB2"""
    # Use pretrained model with transfer learning
    base_model = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)  # Using RGB for transfer learning
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Build classifier on top
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile model with simpler metrics to avoid shape issues
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']  # Start with just accuracy
    )
    
    model.summary()
    return model

def create_callbacks():
    """Create callbacks for model training"""
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',  # Changed from val_auc
        patience=5,
        restore_best_weights=True,
        mode='max'  # Maximize accuracy
    )
    
    # Reduce learning rate when plateau is reached
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Model checkpoint
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_PATH, 'dr_classifier_best.h5'),
        monitor='val_accuracy',  # Changed from val_auc
        save_best_only=True,
        mode='max',  # Maximize accuracy
        verbose=1
    )
    
    # TensorBoard logging
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(MODEL_SAVE_PATH, 'logs'),
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    return [early_stopping, reduce_lr, model_checkpoint, tensorboard]

def train_model_with_fine_tuning(model, train_ds, val_ds, initial_epochs=15, fine_tune_epochs=15, callbacks=None):
    """Train the classifier model with fine-tuning step for transfer learning"""
    print("Training top layers with frozen base model...")
    
    # Train the top layers first
    history_initial = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Unfreeze the base model for fine-tuning
    print("\nFine-tuning the model...")
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Recompile the model with a lower learning rate - use same metrics as initial compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']  # Keep metrics consistent
    )
    
    # Continue training
    history_fine_tune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs + fine_tune_epochs,
        initial_epoch=initial_epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    history = {}
    for key in history_initial.history:
        history[key] = history_initial.history[key] + history_fine_tune.history[key]
    
    return history, model

def evaluate_model(model, test_ds, test_data):
    """Evaluate the model and generate performance reports and visualizations"""
    test_images, test_labels = test_data
    
    # Evaluate model on test set
    print("\nEvaluating model on test data...")
    test_results = model.evaluate(test_ds, verbose=1)
    test_metrics = dict(zip(model.metrics_names, test_results))
    
    print("\nTest metrics:")
    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Get predictions for confusion matrix
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.array([])
    for _, labels in test_ds:
        y_true = np.append(y_true, labels.numpy())
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(NUM_CLASSES), 
                yticklabels=range(NUM_CLASSES))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(RESULTS_PATH, 'confusion_matrix.png'), bbox_inches='tight')
    
    # Generate classification report
    cr = classification_report(y_true, y_pred, target_names=[f'Class {i}' for i in range(NUM_CLASSES)], output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    cr_df.to_csv(os.path.join(RESULTS_PATH, 'classification_report.csv'))
    
    # Save a sample of the test images with predictions for visual inspection
    test_sample_indices = np.random.choice(len(test_images), min(25, len(test_images)), replace=False)
    
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    for i, idx in enumerate(test_sample_indices):
        ax = axes[i//5, i%5]
        
        # Load and resize the image
        img = cv2.imread(test_images[idx], cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, IMAGE_SIZE)
        
        # Process for prediction
        if CHANNELS == 1:
            img_tensor = np.expand_dims(img, axis=-1)
            img_tensor = np.tile(img_tensor, [1, 1, 3])  # Convert to 3 channels for display
        else:
            img_tensor = img
        
        img_tensor = np.expand_dims(img_tensor.astype(np.float32) / 255.0, axis=0)
        
        # Get prediction
        pred = model.predict(img_tensor)[0]
        pred_class = np.argmax(pred)
        true_class = test_labels[idx]
        
        # Display image with prediction
        ax.imshow(img, cmap='gray' if CHANNELS == 1 else None)
        color = 'green' if pred_class == true_class else 'red'
        ax.set_title(f"True: {true_class}\nPred: {pred_class}", color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'sample_predictions.png'), bbox_inches='tight')
    
    return test_metrics, cr_df

def plot_training_history(history):
    """Plot training history metrics"""
    metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 20))
    
    for i, metric in enumerate(metrics):
        axes[i].plot(history[metric], label=f'Training {metric}')
        axes[i].plot(history[f'val_{metric}'], label=f'Validation {metric}')
        axes[i].set_title(f'{metric.capitalize()} over epochs')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'training_history.png'), bbox_inches='tight')
    plt.close()

def compare_with_without_synthetic(initial_epochs=10, fine_tune_epochs=10):
    """Compare model performance with and without synthetic data"""
    # Train with only original data
    print("\n=== Training with original data only ===")
    train_ds_orig, val_ds_orig, test_ds_orig, train_data_orig, val_data_orig, test_data_orig = build_combined_dataset(
        use_synthetic=False
    )
    
    model_orig = build_classifier_model()
    callbacks = create_callbacks()
    
    history_orig, model_orig = train_model_with_fine_tuning(
        model_orig, train_ds_orig, val_ds_orig, 
        initial_epochs=initial_epochs, 
        fine_tune_epochs=fine_tune_epochs, 
        callbacks=callbacks
    )
    
    metrics_orig, cr_orig = evaluate_model(model_orig, test_ds_orig, test_data_orig)
    model_orig.save(os.path.join(MODEL_SAVE_PATH, 'dr_classifier_original_only.h5'))
    
    # Train with combined data (original + synthetic)
    print("\n=== Training with combined data (original + synthetic) ===")
    train_ds_comb, val_ds_comb, test_ds_comb, train_data_comb, val_data_comb, test_data_comb = build_combined_dataset(
        use_synthetic=True, synthetic_ratio=1.0
    )
    
    model_comb = build_classifier_model()
    
    history_comb, model_comb = train_model_with_fine_tuning(
        model_comb, train_ds_comb, val_ds_comb, 
        initial_epochs=initial_epochs, 
        fine_tune_epochs=fine_tune_epochs, 
        callbacks=callbacks
    )
    
    metrics_comb, cr_comb = evaluate_model(model_comb, test_ds_comb, test_data_comb)
    model_comb.save(os.path.join(MODEL_SAVE_PATH, 'dr_classifier_combined.h5'))
    
    # Plot combined history
    plot_training_history(history_orig)
    plot_training_history(history_comb)
    
    # Compare results
    comparison = {
        'Original Only': {
            'accuracy': metrics_orig['accuracy'],
            'auc': metrics_orig['auc'],
            'precision': metrics_orig['precision'],
            'recall': metrics_orig['recall']
        },
        'With Synthetic': {
            'accuracy': metrics_comb['accuracy'],
            'auc': metrics_comb['auc'],
            'precision': metrics_comb['precision'],
            'recall': metrics_comb['recall']
        }
    }
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(os.path.join(RESULTS_PATH, 'model_comparison.csv'))
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    comparison_df.plot(kind='bar')
    plt.title('Performance Comparison: Original vs. With Synthetic Data')
    plt.ylabel('Score')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(RESULTS_PATH, 'model_comparison.png'), bbox_inches='tight')
    plt.close()
    
    return comparison_df

def main():
    """Main function to train and evaluate diabetic retinopathy classifier"""
    print("Starting diabetic retinopathy classifier training")
    
    # Option 1: Train a single model with both original and synthetic data
    train_ds, val_ds, test_ds, train_data, val_data, test_data = build_combined_dataset(
        use_synthetic=True, synthetic_ratio=1.0
    )
    
    model = build_classifier_model()
    callbacks = create_callbacks()
    
    # Train the model
    history, model = train_model_with_fine_tuning(
        model, train_ds, val_ds, 
        initial_epochs=15, 
        fine_tune_epochs=15, 
        callbacks=callbacks
    )
    
    # Evaluate the model
    test_metrics, cr_df = evaluate_model(model, test_ds, test_data)
    
    # Plot training history
    plot_training_history(history)
    
    # Save the final model
    model.save(os.path.join(MODEL_SAVE_PATH, 'dr_classifier_final.h5'))
    
    print("Diabetic retinopathy classifier training and evaluation complete!")
    
    # Option 2: Compare models with and without synthetic data
    # Uncomment to run this comparison (takes longer)
    # compare_with_without_synthetic(initial_epochs=10, fine_tune_epochs=10)

if __name__ == "__main__":
    main()
