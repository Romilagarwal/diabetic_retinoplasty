import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import time

# Configure memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Memory growth enabled for GPU")
    except:
        print("Invalid device or cannot modify virtual devices once initialized")

# Enable mixed precision for better performance on RTX GPUs
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

BASE_PATH = r"E:\diabetic-retinoplasty\blindness"
PROCESSED_IMAGES_DIR = os.path.join(BASE_PATH, "processed", "processed_images")
SYNTHETIC_DATA_DIR = os.path.join(BASE_PATH, "synthetic_data")
IMAGE_SIZE = 128  # Reduced for memory efficiency
BATCH_SIZE = 4  # Small batch size for 4GB VRAM
LATENT_DIM = 100
CLASSES = 5  # 0-4 for DR severity levels

os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)
for i in range(CLASSES):
    os.makedirs(os.path.join(SYNTHETIC_DATA_DIR, str(i)), exist_ok=True)

class DRGAN:
    def __init__(self):
        # Optimizer settings
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        # Loss functions
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        # Build networks
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
    def build_generator(self):
        """Build the DR-GAN++ generator with label conditioning"""
        noise_input = tf.keras.layers.Input(shape=(LATENT_DIM,))
        label_input = tf.keras.layers.Input(shape=(1,))
        
        # Convert label to one-hot encoding
        label_embedding = tf.keras.layers.CategoryEncoding(
            num_tokens=CLASSES, output_mode="one_hot")(label_input)
        
        # Concatenate noise and label
        x = tf.keras.layers.Concatenate()([noise_input, label_embedding])
        
        # Dense layers
        x = tf.keras.layers.Dense(8 * 8 * 256)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Reshape((8, 8, 256))(x)
        
        # Upsampling blocks with residual connections
        x = self.upsample_block(x, 128)  # 16x16
        x = self.upsample_block(x, 64)   # 32x32
        x = self.upsample_block(x, 32)   # 64x64
        x = self.upsample_block(x, 16)   # 128x128
        
        # Output layer - tanh activation for [-1, 1] range
        output_img = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', activation='tanh')(x)
        
        return tf.keras.Model([noise_input, label_input], output_img, name="generator")
    
    def upsample_block(self, x, filters):
        """Upsample block with skip connections for generator"""
        # Store input for residual connection
        residual = x
        
        # First conv
        x = tf.keras.layers.Conv2DTranspose(filters, 3, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        
        # Second conv
        x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        
        # Upsample residual for matching dimensions
        residual = tf.keras.layers.Conv2DTranspose(filters, 1, strides=2, padding='same')(residual)
        
        # Add residual connection
        x = tf.keras.layers.Add()([x, residual])
        
        return x
    
    def build_discriminator(self):
        """Build the DR-GAN++ discriminator with PatchGAN architecture"""
        # Input image
        img_input = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
        
        # Label input
        label_input = tf.keras.layers.Input(shape=(1,))
        
        # Convert label to one-hot encoding
        label_embedding = tf.keras.layers.CategoryEncoding(
            num_tokens=CLASSES, output_mode="one_hot")(label_input)
        
        # Expand label to match image dimensions
        label_channels = tf.keras.layers.Dense(IMAGE_SIZE * IMAGE_SIZE)(label_embedding)
        label_channels = tf.keras.layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 1))(label_channels)
        
        # Concatenate image and label channels
        x = tf.keras.layers.Concatenate()([img_input, label_channels])
        
        # Downsampling blocks
        x = self.downsample_block(x, 32, apply_bn=False)  # 64x64
        x = self.downsample_block(x, 64)                  # 32x32
        x = self.downsample_block(x, 128)                 # 16x16
        x = self.downsample_block(x, 256)                 # 8x8
        
        # PatchGAN output - don't use sigmoid here since we use from_logits=True in loss
        x = tf.keras.layers.Conv2D(1, 4)(x)
        
        return tf.keras.Model([img_input, label_input], x, name="discriminator")
    
    def downsample_block(self, x, filters, apply_bn=True):
        """Downsample block for discriminator"""
        x = tf.keras.layers.Conv2D(filters, 4, strides=2, padding='same')(x)
        if apply_bn:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        return x
    
    def generator_loss(self, fake_output):
        """Loss function for generator"""
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    def discriminator_loss(self, real_output, fake_output):
        """Loss function for discriminator"""
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss
    
    @tf.function
    def train_step(self, real_images, labels):
        """Single training step for GAN"""
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, LATENT_DIM])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate images
            generated_images = self.generator([noise, labels], training=True)
            
            # Train discriminator
            real_output = self.discriminator([real_images, labels], training=True)
            fake_output = self.discriminator([generated_images, labels], training=True)
            
            # Calculate losses
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
        
        # Calculate gradients and apply
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss

def load_dr_dataset_by_class(class_idx, batch_size=BATCH_SIZE):
    """Load diabetic retinopathy images for a specific class"""
    class_dir = os.path.join(PROCESSED_IMAGES_DIR, str(class_idx))
    if not os.path.exists(class_dir):
        print(f"Error: Class directory {class_dir} does not exist")
        return None
    
    # Find image files
    img_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.png')]
    if not img_files:
        print(f"Error: No images found in {class_dir}")
        return None
    
    print(f"Found {len(img_files)} images for class {class_idx}")
    
    def load_and_preprocess(file_path):
        # Read image
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=1)
        
        # Resize
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
        
        # Normalize to [-1, 1]
        img = (tf.cast(img, tf.float32) - 127.5) / 127.5
        
        # Create label tensor
        label = tf.constant([class_idx], dtype=tf.float32)
        
        return img, label
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(img_files)
    dataset = dataset.shuffle(len(img_files))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def train_dr_gan_for_class(class_idx, epochs=100):
    """Train DR-GAN++ for a specific class"""
    class_dataset = load_dr_dataset_by_class(class_idx)
    if class_dataset is None:
        return None
    
    # Create GAN model
    gan = DRGAN()
    
    # Define directories for results
    class_results_dir = os.path.join(SYNTHETIC_DATA_DIR, f"training_results_{class_idx}")
    os.makedirs(class_results_dir, exist_ok=True)
    
    # Determine number of samples per epoch
    dataset_size = sum(1 for _ in class_dataset)
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} for class {class_idx}")
        start_time = time.time()
        
        # Initialize metrics for this epoch
        gen_loss_values = []
        disc_loss_values = []
        
        # Train on batches
        for image_batch, label_batch in tqdm(class_dataset, desc=f"Class {class_idx} - Epoch {epoch+1}"):
            g_loss, d_loss = gan.train_step(image_batch, label_batch)
            gen_loss_values.append(g_loss)
            disc_loss_values.append(d_loss)
        
        # Calculate mean losses
        epoch_g_loss = tf.reduce_mean(gen_loss_values)
        epoch_d_loss = tf.reduce_mean(disc_loss_values)
        
        # Generate sample images every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Generate sample images
            num_samples = min(16, BATCH_SIZE)
            random_noise = tf.random.normal([num_samples, LATENT_DIM])
            labels = tf.constant([[class_idx]] * num_samples, dtype=tf.float32)
            generated_images = gan.generator([random_noise, labels], training=False)
            
            # Rescale images to [0, 1]
            generated_images = (generated_images + 1) / 2.0
            
            # Plot and save
            fig = plt.figure(figsize=(8, 8))
            for i in range(num_samples):
                plt.subplot(4, 4, i+1)
                plt.imshow(generated_images[i, :, :, 0], cmap='gray')
                plt.axis('off')
            
            plt.tight_layout()
            sample_file = os.path.join(class_results_dir, f'samples_epoch_{epoch+1}.png')
            plt.savefig(sample_file)
            plt.close(fig)
        
        time_taken = time.time() - start_time
        print(f"Time for epoch {epoch+1}: {time_taken:.2f} sec")
        print(f"Generator loss: {epoch_g_loss:.4f}, Discriminator loss: {epoch_d_loss:.4f}")
    
    # Save the trained generator
    gan.generator.save(os.path.join(SYNTHETIC_DATA_DIR, f'generator_class_{class_idx}.h5'))
    print(f"Generator for class {class_idx} saved")
    
    return gan

def generate_synthetic_images(class_idx, num_images=100):
    """Generate synthetic images for a specific class using the trained generator"""
    # Load the generator
    generator_path = os.path.join(SYNTHETIC_DATA_DIR, f'generator_class_{class_idx}.h5')
    if not os.path.exists(generator_path):
        print(f"Error: Generator for class {class_idx} not found at {generator_path}")
        return
    
    generator = tf.keras.models.load_model(generator_path)
    print(f"Generator for class {class_idx} loaded")
    
    # Output directory
    output_dir = os.path.join(SYNTHETIC_DATA_DIR, str(class_idx))
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate images in batches
    num_batches = int(np.ceil(num_images / BATCH_SIZE))
    generated_count = 0
    
    for batch in tqdm(range(num_batches), desc=f"Generating class {class_idx} images"):
        current_batch_size = min(BATCH_SIZE, num_images - generated_count)
        if current_batch_size <= 0:
            break
            
        # Generate random noise and labels
        random_noise = tf.random.normal([current_batch_size, LATENT_DIM])
        labels = tf.constant([[class_idx]] * current_batch_size, dtype=tf.float32)
        
        # Generate images
        generated_images = generator([random_noise, labels], training=False)
        
        # Rescale to [0, 1]
        generated_images = (generated_images + 1) / 2.0
        
        # Save each image
        for i in range(current_batch_size):
            # Convert to uint8
            img = (generated_images[i, :, :, 0].numpy() * 255).astype(np.uint8)
            
            # Apply post-processing similar to your original pipeline
            # This helps ensure synthetic images match real image characteristics
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            img = cv2.medianBlur(img, 3)  # Light blur to reduce artifacts
            
            # Save the image
            img_path = os.path.join(output_dir, f'synthetic_{class_idx}_{generated_count + i:04d}.png')
            cv2.imwrite(img_path, img)
        
        generated_count += current_batch_size
    
    print(f"Generated {generated_count} synthetic images for class {class_idx}")
    return output_dir

def analyze_class_distribution():
    """Analyze the class distribution in the dataset"""
    class_counts = []
    
    print("Analyzing class distribution in the dataset:")
    for i in range(CLASSES):
        class_dir = os.path.join(PROCESSED_IMAGES_DIR, str(i))
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
            class_counts.append(count)
            print(f"Class {i}: {count} images")
        else:
            class_counts.append(0)
            print(f"Class {i}: 0 images (directory not found)")
    
    # Calculate ideal balanced distribution
    total_images = sum(class_counts)
    ideal_per_class = total_images / CLASSES
    
    print(f"\nTotal images: {total_images}")
    print(f"Ideal balanced distribution: {ideal_per_class:.1f} images per class")
    
    # Calculate how many synthetic images to generate
    synthetic_counts = []
    for i, count in enumerate(class_counts):
        if count < ideal_per_class:
            synthetic_counts.append(int(ideal_per_class - count))
        else:
            synthetic_counts.append(0)
    
    print("\nSynthetic images needed to balance classes:")
    for i, count in enumerate(synthetic_counts):
        print(f"Class {i}: {count} synthetic images")
    
    return class_counts, synthetic_counts

def balance_dataset():
    """Balance the dataset by generating synthetic images for underrepresented classes"""
    # Analyze current distribution
    class_counts, synthetic_counts = analyze_class_distribution()
    
    # Find underrepresented classes
    underrepresented = [i for i, count in enumerate(synthetic_counts) if count > 0]
    
    if not underrepresented:
        print("Dataset is already balanced. No synthetic data generation needed.")
        return
    
    print(f"\nTraining DR-GAN++ for {len(underrepresented)} underrepresented classes: {underrepresented}")
    
    # Train GAN and generate synthetic images for each underrepresented class
    for class_idx in underrepresented:
        print(f"\n{'='*50}")
        print(f"Processing class {class_idx}")
        print(f"{'='*50}")
        
        # Train GAN
        epochs = 100  # You may need to adjust based on dataset size
        print(f"Training DR-GAN++ for class {class_idx} with {epochs} epochs")
        gan = train_dr_gan_for_class(class_idx, epochs=epochs)
        
        if gan is not None:
            # Generate synthetic images
            num_to_generate = synthetic_counts[class_idx]
            print(f"Generating {num_to_generate} synthetic images for class {class_idx}")
            generate_synthetic_images(class_idx, num_images=num_to_generate)
    
    print("\nDataset balancing complete! Synthetic images are stored in:", SYNTHETIC_DATA_DIR)

if __name__ == "__main__":
    # Limit memory usage for RTX 3050 with 4GB VRAM
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3584)]  # 3.5GB limit
            )
            print("GPU memory limited to 3.5GB")
        except RuntimeError as e:
            print(e)
    
    # Execute dataset balancing
    balance_dataset()