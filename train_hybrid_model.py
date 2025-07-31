import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from swin_transformer import create_hybrid_model
import time
from dotenv import load_dotenv
load_dotenv()

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Memory growth enabled for GPU")
    except:
        print("Failed to enable memory growth")

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print("Mixed precision enabled")

BASE_PATH = os.environ.get('BASE_PATH')
PROCESSED_IMAGES_DIR = os.environ.get('PROCESSED_IMAGES_DIR')
MODEL_SAVE_PATH = os.environ.get('MODEL_SAVE_PATH')
BATCH_SIZE = 4
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 5
EPOCHS = 20
INITIAL_LR = 1e-4
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def focal_loss(gamma=2.0, alpha=0.25):
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
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
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
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
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
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE).cache()
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE).cache()
    return train_ds, val_ds

def compute_class_weights(train_ds):
    labels = []
    for _, y in train_ds.unbatch():
        labels.append(np.argmax(y.numpy()))
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}")
    return class_weight_dict

def train_hybrid_model():
    train_ds, val_ds = load_datasets()
    class_weights = compute_class_weights(train_ds)
    model = create_hybrid_model(input_shape=(224, 224, 1), num_classes=NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    print("Hybrid model created successfully")
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
    print("Starting training...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")
    model.save(os.path.join(MODEL_SAVE_PATH, 'hybrid_model_final.h5'))
    print(f"Model saved to {os.path.join(MODEL_SAVE_PATH, 'hybrid_model_final.h5')}")
    plot_training_history(history)
    return model, history

def plot_training_history(history):
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
