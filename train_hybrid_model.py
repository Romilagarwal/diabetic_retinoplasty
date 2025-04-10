import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from swin_transformer import create_hybrid_model
import time

# Configure memory growth to prevent OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Memory growth enabled for GPU")
    except:
        print("Failed to enable memory growth")

# Enable mixed precision for better performance on RTX GPUs
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print("Mixed precision enabled")

# Constants
BASE_PATH = r"E:\diabetic-retinoplasty\blindness"
PROCESSED_IMAGES_DIR = os.path.join(BASE_PATH, "processed", "processed_images")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "models")
BATCH_SIZE = 4  # Smaller batch size for memory efficiency
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 5
EPOCHS = 20
INITIAL_LR = 1e-4

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for addressing class imbalance
    gamma: focuses more on hard examples
    alpha: addresses class imbalance
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)

        focal_weight = tf.pow(1 - y_pred, gamma) * y_true

        if alpha is not None:
            focal_weight = alpha * focal_weight

        loss = focal_weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    return focal_loss_fixed

def load_datasets():
    """Load preprocessed images from directory structure"""
    print(f"Loading datasets from {PROCESSED_IMAGES_DIR}")

    # Data augmentation for training
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    # Load training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_IMAGES_DIR,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset='training'
    )
    
    # Apply augmentation to training data
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Load validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_IMAGES_DIR,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset='validation'
    )

    # Optimize performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE).cache()
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE).cache()

    return train_ds, val_ds

def compute_class_weights(train_ds):
    """Compute class weights to handle class imbalance"""
    # Extract class labels from dataset
    labels = []
    for _, y in train_ds.unbatch():
        labels.append(np.argmax(y.numpy()))
    
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    # Convert to dictionary format for Keras
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}")
    
    return class_weight_dict

def train_hybrid_model():
    """Train the hybrid EfficientNet+Swin model with focal loss"""
    # Load datasets
    train_ds, val_ds = load_datasets()
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(train_ds)
    
    # Create the hybrid model
    model = create_hybrid_model(input_shape=(224, 224, 1), num_classes=NUM_CLASSES)
    
    # Compile with focal loss to address class imbalance
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss=focal_loss(gamma=2.0, alpha=0.25),  # Use focal loss instead of categorical crossentropy
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    print("Hybrid model created successfully")
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_SAVE_PATH, 'hybrid_model_best.h5'),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train the model
    print("Starting training...")
    start_time = time.time()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,  # Apply class weights to handle imbalance
        callbacks=callbacks
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")
    
    # Save final model
    model.save(os.path.join(MODEL_SAVE_PATH, 'hybrid_model_final.h5'))
    print(f"Model saved to {os.path.join(MODEL_SAVE_PATH, 'hybrid_model_final.h5')}")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc_1'])
    plt.plot(history.history['val_auc_1'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'hybrid_training_history.png'))
    plt.close()

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3584)]
            )
        except RuntimeError as e:
            print(e)
    
    model, history = train_hybrid_model()