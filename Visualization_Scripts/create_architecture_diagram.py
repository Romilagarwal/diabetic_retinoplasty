import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from swin_transformer import create_hybrid_model

def create_detailed_architecture_diagram():
    """Create detailed architecture diagram showing data flow"""
    plt.figure(figsize=(16, 10))

    components = [
        {"name": "Input Image\n224×224×1", "position": [0.1, 0.7, 0.15, 0.15], "color": "lightgray"},
        {"name": "Triplication\n224×224×3", "position": [0.3, 0.7, 0.15, 0.15], "color": "lightblue"},
        {"name": "EfficientNetB0\nFeature Extraction", "position": [0.5, 0.7, 0.15, 0.15], "color": "lightgreen"},
        {"name": "Transformer Block\nWindow Attention", "position": [0.7, 0.7, 0.15, 0.15], "color": "#FFD580"},
        {"name": "Global Average\nPooling", "position": [0.5, 0.4, 0.15, 0.15], "color": "lightcoral"},
        {"name": "Dense Layer\n+ Dropout", "position": [0.7, 0.4, 0.15, 0.15], "color": "#D8BFD8"},
        {"name": "Softmax\nOutput (5 classes)", "position": [0.9, 0.4, 0.15, 0.18], "color": "#87CEFA"}
    ]

    # Draw components
    ax = plt.gca()
    for comp in components:
        rect = plt.Rectangle((comp["position"][0], comp["position"][1]),
                           comp["position"][2], comp["position"][3],
                           facecolor=comp["color"], alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        plt.text(comp["position"][0] + comp["position"][2]/2,
                 comp["position"][1] + comp["position"][3]/2,
                 comp["name"], ha='center', va='center', fontsize=11)

    # Draw arrows
    arrow_style = dict(arrowstyle='->', linewidth=2, color='black')
    ax.annotate('', xy=(0.3, 0.75), xytext=(0.1 + 0.15, 0.75), arrowprops=arrow_style)
    ax.annotate('', xy=(0.5, 0.75), xytext=(0.3 + 0.15, 0.75), arrowprops=arrow_style)
    ax.annotate('', xy=(0.7, 0.75), xytext=(0.5 + 0.15, 0.75), arrowprops=arrow_style)
    ax.annotate('', xy=(0.5, 0.4), xytext=(0.7, 0.55), arrowprops=dict(arrowstyle='->', linewidth=2, color='black', connectionstyle="arc3,rad=-0.3"))
    ax.annotate('', xy=(0.7, 0.5), xytext=(0.5 + 0.15, 0.5), arrowprops=arrow_style)
    ax.annotate('', xy=(0.9, 0.5), xytext=(0.7 + 0.15, 0.5), arrowprops=arrow_style)

    # Add title and labels
    plt.title('Hybrid EfficientNet-B0 + Swin Transformer Architecture for DR Detection', fontsize=20)
    plt.text(0.5, 0.9, 'Feature Extraction', ha='center', fontsize=20)
    plt.text(0.7, 0.25, 'Classification', ha='center', fontsize=20)

    # Remove axis ticks and labels
    plt.axis('off')

    # Save the diagram
    output_dir = os.path.join("E:\\diabetic-retinoplasty\\blindness", "results", "diagrams")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'hybrid_model_architecture.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Architecture diagram saved to {os.path.join(output_dir, 'hybrid_model_architecture.png')}")

if __name__ == "__main__":
    create_detailed_architecture_diagram()
