import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import cv2

BASE_PATH = r"E:\diabetic-retinoplasty\blindness"
GRADCAM_DIR = os.path.join(BASE_PATH, "results", "explainable_ai")
OUTPUT_DIR = os.path.join(BASE_PATH, "results", "explainability_paper_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_gradcam_figure_panels():
    """Create publication-quality figure panels from Grad-CAM visualizations"""
    # Find Grad-CAM images for each class
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    gradcam_files = {}
    for cls in class_names:
        files = glob(os.path.join(GRADCAM_DIR, f"*_{cls}.png"))
        if files:
            gradcam_files[cls] = files[:3]  # Take up to 3 examples per class
    
    # Create figure panels for each class
    for cls, files in gradcam_files.items():
        if not files:
            continue
        
        fig, axes = plt.subplots(len(files), 3, figsize=(15, 5*len(files)))
        
        # Handle the case where there's only one file (one row)
        if len(files) == 1:
            axes = [axes]  # Convert to list of rows
        
        for i, file_path in enumerate(files):
            # Load the GradCAM visualization
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # The GradCAM image has 3 panels: original, heatmap, overlay
            # Extract and display each panel
            height, width = img.shape[:2]
            panel_width = width // 3
            
            original = img[:, :panel_width]
            heatmap = img[:, panel_width:2*panel_width]
            overlay = img[:, 2*panel_width:]
            
            axes[i][0].imshow(original)
            axes[i][0].set_title(f'Original Image {i+1}')
            axes[i][0].axis('off')
            
            axes[i][1].imshow(heatmap)
            axes[i][1].set_title(f'Activation Heatmap {i+1}')
            axes[i][1].axis('off')
            
            axes[i][2].imshow(overlay)
            axes[i][2].set_title(f'Overlay {i+1}')
            axes[i][2].axis('off')
        
        plt.suptitle(f'Grad-CAM Visualizations for {cls} Class', fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(OUTPUT_DIR, f'gradcam_panel_{cls.lower().replace(" ", "_")}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

def create_explainability_summary():
    """Create a summary figure highlighting key findings from explainability analysis"""
    # Find one example for each class for a summary panel
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    examples = []
    for cls in class_names:
        files = glob(os.path.join(GRADCAM_DIR, f"*_{cls}.png"))
        if files:
            examples.append((cls, files[0]))
    
    if not examples:
        print("No Grad-CAM examples found for summary")
        return
    
    # Create a summary panel with one row per class
    fig, axes = plt.subplots(len(examples), 3, figsize=(15, 5*len(examples)))
    
    # Fix for single row case
    if len(examples) == 1:
        axes = [axes]  # Convert to list of rows when only one row exists
    
    for i, (cls, file_path) in enumerate(examples):
        # Load the GradCAM visualization
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract panels
        height, width = img.shape[:2]
        panel_width = width // 3
        
        original = img[:, :panel_width]
        heatmap = img[:, panel_width:2*panel_width]
        overlay = img[:, 2*panel_width:]
        
        axes[i][0].imshow(original)
        axes[i][0].set_title(f'Original - {cls}')
        axes[i][0].axis('off')
        
        axes[i][1].imshow(heatmap)
        axes[i][1].set_title(f'Heatmap - {cls}')
        axes[i][1].axis('off')
        
        axes[i][2].imshow(overlay)
        axes[i][2].set_title(f'Overlay - {cls}')
        axes[i][2].axis('off')
    
    plt.suptitle('Explainability Analysis Across Diabetic Retinopathy Severity Classes', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, 'explainability_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_explainability_visualizations():
    """Create all explainability visualizations"""
    if not os.path.exists(GRADCAM_DIR):
        print(f"Error: Grad-CAM directory not found at {GRADCAM_DIR}")
        return
    
    create_gradcam_figure_panels()
    create_explainability_summary()
    
    print(f"Explainability visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    create_explainability_visualizations()