# Diabetic Retinopathy Detection

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9%2B-FF6F00?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-AI-9cf?style=flat-square&logo=pytorch)](https://github.com/yourusername/diabetic-retinopathy-detection)

A hybrid deep learning framework for automated diabetic retinopathy detection with uncertainty quantification and explainability.

![Diabetic Retinopathy System](assets/reseach_summary_dashboard.png)

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Future Work](#-future-work)
- [References](#-references)
- [Contributing](#-contributing)

## 🔍 Overview

Diabetic Retinopathy (DR) is a diabetes complication that affects the eyes and can lead to blindness if left untreated. Early detection is crucial for effective treatment, but manual screening by ophthalmologists is time-consuming and subject to variability.

This project implements a hybrid deep learning approach that:

1. **Accurately classifies** retinal images into 5 severity levels of DR
2. **Quantifies uncertainty** in predictions using Bayesian methods
3. **Explains decisions** through gradient-based visualization techniques
4. **Addresses class imbalance** through focal loss and weighting strategies

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Preprocessing Pipeline** | Ben Graham's technique with green channel extraction, CLAHE, and denoising |
| 🧠 **Hybrid Architecture** | EfficientNetB0 + Swin Transformer for improved feature representation |
| 🔍 **Bayesian Uncertainty** | Monte Carlo Dropout for confidence estimation and uncertainty quantification |
| 👁️ **Explainable AI** | Grad-CAM visualizations showing which retinal regions influence decisions |
| ⚖️ **Class Imbalance Handling** | Focal Loss and class weighting techniques to handle unbalanced datasets |

## 🏗️ Model Architecture

![Architecture](assets\hybrid_model_architecture.png)

Our hybrid architecture combines:
- **EfficientNetB0**: Pre-trained CNN for efficient feature extraction
- **Swin Transformer**: Attention-based refinement of features with hierarchical window partitioning
- **Monte Carlo Dropout**: Bayesian approximation for uncertainty estimation
- **Grad-CAM**: Class activation mapping for model explainability

## 📊 Dataset

The model is trained and evaluated on the [APTOS 2019 Diabetic Retinopathy Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) dataset, which contains retinal fundus photographs labeled with DR severity levels:

| Class | Severity Level | Description | Visual Signs |
|-------|---------------|-------------|--------------|
| 0 | No DR | No signs of diabetic retinopathy | Normal retina |
| 1 | Mild | Microaneurysms only | Small red dots |
| 2 | Moderate | More than microaneurysms but less than severe | Red lesions, hard exudates |
| 3 | Severe | Extensive hemorrhages and venous beading | Cotton wool spots, venous beading |
| 4 | Proliferative | Abnormal blood vessel growth and potential retinal detachment | Neovascularization, preretinal hemorrhage |

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/romilagarwal/diabetic_retinoplasty.git
cd diabetic_retinopathy

# Create and activate virtual environment
python -m venv env
source env/bin/activate  
# On Windows use: env\Scripts\activate

# Install dependencies from requirements
pip install -r [requirements.txt]
```
Additionally, Graphviz must be installed on your system for model visualization.


## 🚀 Usage

1. Data Preprocessing
```bash
python pre_process_with_dataset_download.py
```
<details> <summary><b>Preprocessing Details</b></summary>
This script:

Downloads the dataset (if not already present)
Applies Ben Graham's preprocessing technique with green channel extraction
Enhances images with CLAHE (Contrast Limited Adaptive Histogram Equalization)
Applies denoising filters
Resizes images to 224×224
Organizes processed images into class folders
</details>

2. Base Model Training
```bash
python efficientnet_model.py
```
Trains a baseline EfficientNetB0 model with transfer learning from ImageNet weights.

3. Hybrid Model Training
```bash
python train_hybrid_model.py
```
<details> <summary><b>Training Parameters</b></summary>
The hybrid model training uses:

 · Focal loss for class imbalance
 · Mixed precision for memory efficiency
 · Class weighting for balanced learning
 · Learning rate scheduling
 · Early stopping to prevent overfitting
</details>

4. Model Evaluation
```bash
# Test the base model
python testing_efficientnet_model.py

# Test the hybrid model
python test_hybrid_model.py
```

5.Bayesian Uncertainty Estimation
```bash
python bayesian_inference.py
```
<details> <summary><b>Uncertainty Metrics</b></summary>
The Bayesian component performs:

 · Monte Carlo Dropout inference with multiple forward passes
 · Confidence score calculation
 · Uncertainty estimation (standard deviation of predictions)
 · Predictive entropy calculation
 · Reliability diagram generation
</details>

6.Explainable AI Visualizations
```bash
python explainable_ai.py
```
Generates Generates Grad-CAM visualizations highlighting regions that influence the model's decisions.

7. Generate Visualizations for Publication
```bash
python generate_all_visualizations.py
```
Creates comprehensive visualizations for research papers or presentations.

## 📈 Results
Performance Metrics
|Model        | No DR  | Mild	| Moderate | Severe | Proliferative | Average |
|-------------|--------|--------|----------|--------|---------------|---------|
|EfficientNet | 0.76   | 0.70	|  0.72	   |  0.65	|     0.63	    |   0.69  |
|Hybrid Model |	0.82   | 0.75	|  0.79	   |  0.73	|     0.71	    |   0.76  |

Key Improvements
· +7% Average F1 Score improvement over baseline EfficientNet
· Better Generalization across all DR severity classes
· Enhanced Performance on minority classes (Severe and Proliferative)
· Reduced Uncertainty in predictions compared to baseline

## 👁️ Visualizations

<h3>Grad-CAM Explainability</h3>
<img alt="Grad-CAM" src="assets\explainability_summary.png">

<h3>Uncertainty Analysis</h3>

<img alt="Uncertainty" src="assets\uncertainty_processed.png">


## 🔮 Future Work

1. DR-GAN++: Implementation of Generative Adversarial Networks for synthetic data generation to further address class imbalance
2. Ensemble Methods: Combining multiple models for improved performance
3. Clinical Integration: Development of a user-friendly interface for clinical use
4. Mobile Deployment: Optimization for edge devices to enable screening in remote areas
Multimodal Learning: Integrating patient metadata with retinal images


## 📚 References

1. Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. Proceedings of the IEEE conference on computer vision and pattern recognition, 4700-4708.

2. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. Proceedings of the IEEE/CVF International Conference on Computer Vision, 10012-10022.

3. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. International conference on machine learning, 1050-1059.

4. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. Proceedings of the IEEE international conference on computer vision, 618-626.

5. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. Proceedings of the IEEE international conference on computer vision, 2980-2988.


## 👥 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
