import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

from swin_transformer import (
    PatchEmbed, WindowAttention, SwinTransformerBlock, 
    MLP, PatchMerging, BasicLayer, SwinTransformer,
    window_partition, window_reverse
)
from test_hybrid_model import Cast  # Import the Cast layer definition

# Configure memory growth to prevent OOM errors
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
RESULTS_DIR = os.path.join(BASE_PATH, "results", "explainable_ai")
os.makedirs(RESULTS_DIR, exist_ok=True)

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

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for showing model focus areas"""
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradient of the predicted class with respect to feature map activations
    with tf.GradientTape() as tape:
        # Add a batch dimension
        img_array = tf.cast(img_array, tf.float32)
        last_conv_layer_output, preds = grad_model(img_array)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        # Use the target class's prediction
        class_channel = preds[:, pred_index]

    # Gradient of the class with respect to the feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Pool the gradients across the channels
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps with the gradients
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purposes, normalize between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.maximum(tf.reduce_max(heatmap), 1e-10)
    
    return heatmap.numpy()

def create_gradcam_overlay(img_path, heatmap, alpha=0.4):
    """Create overlay of heatmap on original image"""
    # Load the original image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=-1)  # Convert to 3 channel for visualization
    img = np.concatenate([img, img, img], axis=-1)
    
    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Apply jet colormap to heatmap
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Create RGB heatmap image
    jet_heatmap = jet_heatmap * 255
    jet_heatmap = np.uint8(jet_heatmap)
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    
    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(img, 1-alpha, jet_heatmap, alpha, 0)
    
    return img, jet_heatmap, superimposed_img

def visualize_gradcam(original_img, heatmap, superimposed_img, class_names, pred_class, output_path):
    """Visualize Grad-CAM results"""
    plt.figure(figsize=(16, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Grad-CAM Heatmap')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title(f'Prediction: {class_names[pred_class]}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def process_test_images(model, test_dir, results_dir, last_conv_layer='swin_refine', max_images=None):
    """Process test images and generate Grad-CAM visualizations"""
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} does not exist!")
        return

    test_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    
    if not test_images:
        print(f"Error: No PNG images found in {test_dir}")
        return
    
    print(f"Found {len(test_images)} test images")
    
    if max_images is not None and max_images > 0:
        test_images = test_images[:max_images]
        print(f"Processing {len(test_images)} images (limited by max_images={max_images})")
    
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    for img_file in tqdm(test_images, desc="Processing images with Grad-CAM"):
        img_path = os.path.join(test_dir, img_file)
        
        try:
            # Read and preprocess image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error: Could not read image {img_path}")
                continue
            
            img = cv2.resize(img, (224, 224))
            img_normalized = img / 255.0
            img_normalized = np.expand_dims(img_normalized, axis=0)
            img_normalized = np.expand_dims(img_normalized, axis=-1)
            
            # Make predictions
            preds = model.predict(img_normalized, verbose=0)
            pred_class_idx = np.argmax(preds[0])
            pred_class_name = class_names[pred_class_idx]
            
            # Generate Grad-CAM
            heatmap = make_gradcam_heatmap(img_normalized, model, last_conv_layer, pred_class_idx)
            
            # Create visualization
            original, heatmap_viz, superimposed = create_gradcam_overlay(img_path, heatmap)
            
            # Save results
            output_path = os.path.join(results_dir, f"gradcam_{os.path.splitext(img_file)[0]}_{pred_class_name}.png")
            visualize_gradcam(original, heatmap_viz, superimposed, class_names, pred_class_idx, output_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Grad-CAM processing complete. Results saved to {results_dir}")

def main():
    # Load the hybrid model
    model = load_model()
    
    # Process images with Grad-CAM
    # Note: Using 'swin_refine' as the last layer because that's the Swin Transformer block
    # before your classification head. This will show what regions the transformer is focusing on.
    process_test_images(model, PROCESSED_TEST_DIR, RESULTS_DIR, 'swin_refine', max_images=None)
    
if __name__ == "__main__":
    main()