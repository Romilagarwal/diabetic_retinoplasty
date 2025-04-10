import os
import tensorflow as tf
import keras
from keras import layers, Model, Input
from keras.applications import EfficientNetB0
import matplotlib.pyplot as plt

BASE_PATH = r"E:\diabetic-retinoplasty\blindness"
PROCESSED_IMAGES_DIR = os.path.join(BASE_PATH, "processed", "processed_images")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "models")
BATCH_SIZE = 8
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 5

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def load_datasets():
    """Load preprocessed images from directory structure"""
    print(f"Loading datasets from {PROCESSED_IMAGES_DIR}")

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

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

def create_efficientnetb0_model(input_shape=(224, 224, 1), num_classes=5):
    """Create EfficientNetB0 model for DR classification"""
    inputs = Input(shape=input_shape)

    x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])

    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    for layer in base_model.layers[:100]:
        layer.trainable = False

    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    return model

def train_model(model, train_ds, val_ds, epochs=30):
    """Train the model with callbacks"""
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_SAVE_PATH, 'efficientnet_best.h5'),
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

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    return history, model

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'training_history.png'))
    plt.close()

def main():
    """Main function to run the EfficientNetB0 implementation"""
    train_ds, val_ds = load_datasets()

    print(f"Train dataset: {train_ds}")
    print(f"Validation dataset: {val_ds}")

    model = create_efficientnetb0_model()
    print("Model created successfully")

    model.summary()

    print("Starting model training...")
    history, trained_model = train_model(model, train_ds, val_ds)

    plot_training_history(history)

    trained_model.save(os.path.join(MODEL_SAVE_PATH, 'efficientnet_final.h5'))
    print(f"Model saved to {os.path.join(MODEL_SAVE_PATH, 'efficientnet_final.h5')}")

    return trained_model

if __name__ == "__main__":
    main()
