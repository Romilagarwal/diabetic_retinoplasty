import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split

# Configure GPU memory growth to prevent OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Memory growth enabled for GPU")
    except:
        print("Failed to enable memory growth")

# Constants and paths (following your project structure)
BASE_PATH = r"E:\diabetic-retinoplasty\blindness"
PROCESSED_IMAGES_DIR = os.path.join(BASE_PATH, "processed", "processed_images")
GAN_IMAGES_DIR = os.path.join(BASE_PATH, "processed", "gan_synthetic_images")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "models")
LATENT_DIM = 128
IMAGE_SIZE = (224, 224)
CHANNELS = 1  # Grayscale images
NUM_CLASSES = 5  # 0-4 DR severity classes
BATCH_SIZE = 8
EPOCHS = 20

# Create necessary directories
os.makedirs(GAN_IMAGES_DIR, exist_ok=True)
for i in range(NUM_CLASSES):
    os.makedirs(os.path.join(GAN_IMAGES_DIR, str(i)), exist_ok=True)

class ConditionalGAN:
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_shape = (*IMAGE_SIZE, CHANNELS)
        
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # For metrics tracking
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
    
    def build_generator(self):
        """Build the conditional generator network for DR-GAN++"""
        # Noise input
        noise = layers.Input(shape=(self.latent_dim,), name="noise_input")
        
        # Class label input
        label = layers.Input(shape=(1,), name="class_label", dtype=tf.int32)
        
        # Embedding for categorical input
        label_embedding = layers.Embedding(self.num_classes, 50)(label)
        label_embedding = layers.Flatten()(label_embedding)
        label_embedding = layers.Dense(8 * 8)(label_embedding)
        
        # Process noise input
        x = layers.Dense(8 * 8 * 256, use_bias=False)(noise)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Reshape((8, 8, 256))(x)
        
        # Combine noise and label
        label_embedding = layers.Reshape((8, 8, 1))(label_embedding)
        x = layers.Concatenate()([x, label_embedding])
        
        # Upsampling blocks
        # 8x8 -> 16x16
        x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # 16x16 -> 32x32
        x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # 32x32 -> 64x64
        x = layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # 64x64 -> 128x128
        x = layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # 128x128 -> 224x224 (adjusting for desired size)
        x = layers.Conv2DTranspose(8, (5, 5), strides=(1.75, 1.75), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Final layer with tanh activation
        output_img = layers.Conv2D(CHANNELS, (5, 5), padding='same', activation='tanh')(x)
        
        # Ensure output size is exactly 224x224
        output_img = layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1])(output_img)
        
        # Create the generator model
        generator = keras.Model([noise, label], output_img, name="dr_generator")
        generator.summary()
        return generator
    
    def build_discriminator(self):
        """Build the conditional discriminator network for DR-GAN++"""
        # Image input
        img_input = layers.Input(shape=self.img_shape, name="image_input")
        
        # Class label input
        label = layers.Input(shape=(1,), name="class_label", dtype=tf.int32)
        
        # Embedding for categorical input
        label_embedding = layers.Embedding(self.num_classes, 50)(label)
        label_embedding = layers.Dense(IMAGE_SIZE[0] * IMAGE_SIZE[1])(label_embedding)
        label_embedding = layers.Reshape((IMAGE_SIZE[0], IMAGE_SIZE[1], 1))(label_embedding)
        
        # Combine image and label
        x = layers.Concatenate()([img_input, label_embedding])
        
        # Convolutional blocks
        x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Final classification
        x = layers.Flatten()(x)
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        # Create the discriminator model
        discriminator = keras.Model([img_input, label], output, name="dr_discriminator")
        discriminator.summary()
        return discriminator
    
    @tf.function
    def train_step(self, real_images, real_labels):
        """Single training step with automatic mixed precision"""
        batch_size = tf.shape(real_images)[0]
        
        # Generate random noise and random labels for fake images
        noise = tf.random.normal([batch_size, self.latent_dim])
        gen_labels = tf.random.uniform([batch_size], minval=0, maxval=self.num_classes, dtype=tf.int32)
        
        with tf.GradientTape() as disc_tape:
            # Generate fake images
            fake_images = self.generator([noise, gen_labels], training=True)
            
            # Get discriminator predictions for real and fake images
            real_output = self.discriminator([real_images, real_labels], training=True)
            fake_output = self.discriminator([fake_images, gen_labels], training=True)
            
            # Calculate discriminator loss
            d_loss_real = keras.losses.BinaryCrossentropy(from_logits=False)(
                tf.ones_like(real_output) * 0.9, real_output)  # Label smoothing
            d_loss_fake = keras.losses.BinaryCrossentropy(from_logits=False)(
                tf.zeros_like(fake_output), fake_output)
            d_loss = d_loss_real + d_loss_fake
        
        # Calculate and apply gradients for discriminator
        d_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )
        
        # Train the generator (generate new noise)
        noise = tf.random.normal([batch_size, self.latent_dim])
        # Generate random labels for new fake images
        gen_labels = tf.random.uniform([batch_size], minval=0, maxval=self.num_classes, dtype=tf.int32)
        
        with tf.GradientTape() as gen_tape:
            # Generate fake images
            fake_images = self.generator([noise, gen_labels], training=True)
            
            # Get discriminator predictions for fake images
            fake_output = self.discriminator([fake_images, gen_labels], training=True)
            
            # Calculate generator loss (want discriminator to think these are real)
            g_loss = keras.losses.BinaryCrossentropy(from_logits=False)(
                tf.ones_like(fake_output), fake_output)
        
        # Calculate and apply gradients for generator
        g_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        
        # Update metrics
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }
    
    def generate_and_save_images(self, epoch, num_examples_per_class=2):
        """Generate and save images for visualization during training"""
        # Create directory for this epoch's samples
        samples_dir = os.path.join(MODEL_SAVE_PATH, f"gan_samples_epoch_{epoch}")
        os.makedirs(samples_dir, exist_ok=True)
        
        rows, cols = NUM_CLASSES, num_examples_per_class
        fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        
        for class_idx in range(NUM_CLASSES):
            # Generate fixed noise for this class
            noise = tf.random.normal([num_examples_per_class, self.latent_dim])
            labels = tf.ones([num_examples_per_class], dtype=tf.int32) * class_idx
            
            # Generate images
            gen_images = self.generator([noise, labels], training=False)
            
            # Scale images to [0, 1]
            gen_images = (gen_images + 1) / 2.0
            
            # Plot and save individual images
            for i in range(num_examples_per_class):
                if rows == 1:
                    ax = axs[i]
                else:
                    ax = axs[class_idx, i]
                
                # Display grayscale image
                ax.imshow(gen_images[i, :, :, 0], cmap='gray')
                ax.set_title(f"Class {class_idx}")
                ax.axis('off')
                
                # Save individual image
                img_array = (gen_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
                img_filename = os.path.join(samples_dir, f"class_{class_idx}_sample_{i}.png")
                cv2.imwrite(img_filename, img_array)
        
        # Save the entire figure
        fig.tight_layout()
        fig_filename = os.path.join(samples_dir, f"gan_samples_epoch_{epoch}.png")
        plt.savefig(fig_filename)
        plt.close(fig)
    
    def save_models(self):
        """Save the generator and discriminator models"""
        self.generator.save(os.path.join(MODEL_SAVE_PATH, "dr_gan_generator.h5"))
        self.discriminator.save(os.path.join(MODEL_SAVE_PATH, "dr_gan_discriminator.h5"))
        print(f"Models saved to {MODEL_SAVE_PATH}")

def load_and_prepare_data(batch_size=BATCH_SIZE):
    """Load and prepare the diabetic retinopathy dataset for GAN training"""
    # Create dataset directory iterators
    class_dirs = [os.path.join(PROCESSED_IMAGES_DIR, str(i)) for i in range(NUM_CLASSES)]
    
    # Check directory structure
    all_images = []
    all_labels = []
    
    for label, directory in enumerate(class_dirs):
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist!")
            continue
        
        image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
        print(f"Class {label}: Found {len(image_files)} images")
        
        for img_file in image_files:
            all_images.append(os.path.join(directory, img_file))
            all_labels.append(label)
    
    # Create train-test split
    train_images, val_images, train_labels, val_labels = train_test_split(
        all_images, all_labels, test_size=0.1, stratify=all_labels, random_state=42
    )
    
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    
    # Create tf.data.Dataset
    def preprocess_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=CHANNELS)
        img = tf.image.resize(img, IMAGE_SIZE)
        # Normalize to [-1, 1] for GAN
        img = (tf.cast(img, tf.float32) - 127.5) / 127.5
        return img
    
    def load_and_preprocess(img_path, label):
        return preprocess_image(img_path), label
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def train_gan(gan, train_ds, epochs=EPOCHS):
    """Train the GAN model"""
    history = {'g_loss': [], 'd_loss': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Reset metrics at the start of each epoch
        gan.gen_loss_tracker.reset_states()
        gan.disc_loss_tracker.reset_states()
        
        # Train the model
        for images, labels in tqdm(train_ds, desc=f"Epoch {epoch+1}"):
            metrics = gan.train_step(images, labels)
        
        # Record metrics
        g_loss = gan.gen_loss_tracker.result().numpy()
        d_loss = gan.disc_loss_tracker.result().numpy()
        history['g_loss'].append(g_loss)
        history['d_loss'].append(d_loss)
        
        print(f"Generator loss: {g_loss:.4f}, Discriminator loss: {d_loss:.4f}")
        
        # Generate and save sample images every 5 epochs or on the last epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            gan.generate_and_save_images(epoch + 1)
    
    # Save the models after training
    gan.save_models()
    
    # Plot and save the learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['g_loss'], label='Generator')
    plt.plot(history['d_loss'], label='Discriminator')
    plt.title('GAN Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'dr_gan_training_history.png'))
    plt.close()
    
    return history

def generate_synthetic_images(num_images_per_class=100):
    """Generate synthetic images to balance the dataset"""
    # Load the trained generator
    generator = tf.keras.models.load_model(os.path.join(MODEL_SAVE_PATH, "dr_gan_generator.h5"))
    
    # Find class frequencies in original dataset
    class_counts = {}
    for i in range(NUM_CLASSES):
        class_dir = os.path.join(PROCESSED_IMAGES_DIR, str(i))
        if os.path.exists(class_dir):
            class_counts[i] = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
        else:
            class_counts[i] = 0
    
    print("Original class distribution:")
    for cls, count in class_counts.items():
        print(f"Class {cls}: {count} images")
    
    # Find maximum class count to balance classes
    max_count = max(class_counts.values())
    
    # Generate synthetic images for underrepresented classes
    for class_idx in range(NUM_CLASSES):
        synthetic_count = max(0, max_count - class_counts[class_idx])
        
        if synthetic_count > 0:
            print(f"Generating {synthetic_count} synthetic images for class {class_idx}")
            
            # Generate in batches to avoid memory issues
            batch_size = min(BATCH_SIZE, synthetic_count)
            remaining = synthetic_count
            
            for batch_idx in tqdm(range(0, synthetic_count, batch_size), 
                                 desc=f"Generating class {class_idx}"):
                # Calculate actual batch size for last batch
                actual_batch = min(batch_size, remaining)
                
                # Generate noise and labels
                noise = tf.random.normal([actual_batch, LATENT_DIM])
                labels = tf.ones([actual_batch], dtype=tf.int32) * class_idx
                
                # Generate images
                gen_images = generator([noise, labels], training=False)
                
                # Scale from [-1, 1] to [0, 255]
                gen_images = ((gen_images + 1) / 2.0 * 255).numpy().astype(np.uint8)
                
                # Save images
                for i in range(actual_batch):
                    img_array = gen_images[i, :, :, 0]
                    img_filename = os.path.join(
                        GAN_IMAGES_DIR, str(class_idx), 
                        f"synthetic_{batch_idx + i:04d}.png"
                    )
                    cv2.imwrite(img_filename, img_array)
                
                remaining -= actual_batch
    
    # Print new distribution with synthetic data
    new_class_counts = {}
    for i in range(NUM_CLASSES):
        original_dir = os.path.join(PROCESSED_IMAGES_DIR, str(i))
        synthetic_dir = os.path.join(GAN_IMAGES_DIR, str(i))
        
        original_count = 0
        if os.path.exists(original_dir):
            original_count = len([f for f in os.listdir(original_dir) if f.endswith('.png')])
            
        synthetic_count = 0
        if os.path.exists(synthetic_dir):
            synthetic_count = len([f for f in os.listdir(synthetic_dir) if f.endswith('.png')])
            
        new_class_counts[i] = (original_count, synthetic_count, original_count + synthetic_count)
    
    print("\nClass distribution after synthetic data generation:")
    print(f"{'Class':>5} | {'Original':>8} | {'Synthetic':>9} | {'Total':>5}")
    print("-" * 40)
    for cls, (orig, synth, total) in new_class_counts.items():
        print(f"{cls:5d} | {orig:8d} | {synth:9d} | {total:5d}")

def main():
    """Main function to run DR-GAN++"""
    print("Initializing DR-GAN++ for diabetic retinopathy synthetic data generation")
    
    # Load and prepare data
    train_ds, val_ds = load_and_prepare_data()
    
    # Create GAN model
    gan = ConditionalGAN(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)
    
    # Train the GAN
    history = train_gan(gan, train_ds, epochs=EPOCHS)
    
    # Generate synthetic images to balance dataset
    generate_synthetic_images()
    
    print("DR-GAN++ training and synthetic data generation complete!")

if __name__ == "__main__":
    # Set memory growth before executing
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")
    
    main()