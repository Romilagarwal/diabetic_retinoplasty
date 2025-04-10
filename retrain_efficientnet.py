import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model

BASE_PATH = r"E:\diabetic-retinoplasty\blindness"
PROCESSED_IMAGES_DIR = os.path.join(BASE_PATH, "processed", "processed_images")
MODEL_PATH = os.path.join(BASE_PATH, "models", "efficientnet_best.h5")
RETRAINED_MODEL_PATH = os.path.join(BASE_PATH, "models", "efficientnet_balanced.h5")
BATCH_SIZE = 8
EPOCHS = 10
TARGET_SIZE = (224, 224)

os.makedirs(os.path.dirname(RETRAINED_MODEL_PATH), exist_ok=True)

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

def load_dataset():
    """Load the preprocessed images with balanced class distribution"""
    print(f"Loading dataset from {PROCESSED_IMAGES_DIR}")

    class_dirs = [os.path.join(PROCESSED_IMAGES_DIR, d) for d in os.listdir(PROCESSED_IMAGES_DIR)
                 if os.path.isdir(os.path.join(PROCESSED_IMAGES_DIR, d))]

    class_counts = []
    for class_dir in class_dirs:
        count = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
        class_counts.append(count)
        print(f"Class {os.path.basename(class_dir)}: {count} images")

    total_samples = sum(class_counts)
    class_weights = {i: total_samples / (len(class_counts) * count)
                    for i, count in enumerate(class_counts)}
    print(f"Class weights: {class_weights}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_IMAGES_DIR,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        image_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset='training'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_IMAGES_DIR,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        image_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset='validation'
    )

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_weights

def retrain_model():
    """Retrain the model with balanced approach"""
    train_ds, val_ds, class_weights = load_dataset()

    print(f"Loading model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            RETRAINED_MODEL_PATH,
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

    print("Retraining model with balanced approach...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, "models", "balanced_training_history.png"))
    plt.close()

    print(f"Model retrained and saved to {RETRAINED_MODEL_PATH}")
    return model

if __name__ == "__main__":
    retrained_model = retrain_model()
