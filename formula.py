import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import textwrap

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'

BASE_PATH = r"C:/Users/romil/projects/diabetic_retinoplasty"
FORMULA_PATH = os.path.join(BASE_PATH, "formula")
os.makedirs(FORMULA_PATH, exist_ok=True)

def create_equation_image(eq, description, filename, figsize=(10, 4), desc_above=False):
    fig = plt.figure(figsize=figsize)
    if description and desc_above:
        wrapped_text = textwrap.fill(description, width=80)
        plt.figtext(0.5, 0.7, wrapped_text, fontsize=12, ha='center', va='bottom')
        plt.figtext(0.5, 0.4, f"${eq}$", fontsize=16, ha='center')
    elif description:
        plt.figtext(0.5, 0.7, f"${eq}$", fontsize=16, ha='center')
        wrapped_text = textwrap.fill(description, width=80)
        plt.figtext(0.5, 0.4, wrapped_text, fontsize=12, ha='center', va='top')
    else:
        plt.figtext(0.5, 0.5, f"${eq}$", fontsize=16, ha='center')
    plt.axis('off')
    plt.savefig(os.path.join(FORMULA_PATH, f"{filename}.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

equations = [
    # 1. Focal Loss
    (r'L_{\text{focal}}(p, y) = -\alpha \sum_{c=1}^{C} y_{c} \bigl(1 - p_{c}\bigr)^{\gamma}\log\bigl(p_{c}\bigr)',
     r'where: $(y_{c})$ is the one-hot label for class $(c)$, $(p_{c} = \hat{y}_c)$ is the predicted probability for class $(c)$, $(\alpha)$ addresses class imbalance, $(\gamma)$ focuses training on hard misclassified examples.',
     "focal_loss", False),

    # 2. Standard Cross-Entropy Loss
    (r'L_{\text{CE}}(p, y) = - \sum_{c=1}^{C} y_{c} \log\bigl(p_{c}\bigr)',
     "",
     "cross_entropy_loss", False),

    # 3. Monte Carlo Dropout Prediction (description above)
    (r'\hat{p}(y = c \mid \mathbf{x}) \approx \frac{1}{T}\sum_{t=1}^{T} p(y = c \mid \mathbf{x}, \boldsymbol{\theta}_t)',
     r'When performing $T$ forward passes (with dropout active at inference), your final probability estimate for each class $c$ is the average.',
     "mc_dropout", True),

    # 4. Uncertainty via Standard Deviation
    (r'\text{Uncertainty}(c) = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \Bigl[p(y=c \mid \mathbf{x}, \boldsymbol{\theta}_t) - \hat{p}(y = c \mid \mathbf{x})\Bigr]^2}',
     "",
     "uncertainty", False),

    # 5. Predictive Entropy (description above)
    (r'H\bigl[p(y \mid \mathbf{x})\bigr] = -\sum_{c=1}^{C} p(y = c \mid \mathbf{x}) \log\bigl[p(y = c \mid \mathbf{x})\bigr]',
     r'A commonly used scalar measure of overall uncertainty is predictive entropy.',
     "predictive_entropy", True),

    # 6. Grad-CAM (description above)
    (r'\alpha_k^c = \frac{1}{Z} \sum_{i}\sum_{j} \frac{\partial y^c}{\partial A_{ij}^k}, \quad L^c = \text{ReLU}\Bigl(\sum_{k} \alpha_k^c A^k\Bigr)',
     r'Grad-CAM highlights the important regions in the feature map for a target class $c$. Suppose $A^k$ is the activation of the $k$-th feature map in some convolutional layer. The gradient $\tfrac{\partial y^c}{\partial A^k}$ is global-average-pooled over spatial locations to get $\alpha_k^c$. Then the Grad-CAM heatmap $L^c$ is computed as shown above, where $Z$ is a normalization factor (often the spatial dimension of the feature map), $\text{ReLU}(\cdot)$ is the rectified linear unit, and $y^c$ is the model\'s output score (logit) for class $c$.',
     "grad_cam", True)
]

for eq, description, filename, desc_above in equations:
    height = 4 if description else 2
    create_equation_image(eq, description, filename, figsize=(10, height), desc_above=desc_above)

print("All equation images generated in the 'formula' folder")
