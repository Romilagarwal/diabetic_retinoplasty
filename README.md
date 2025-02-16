Diabetic retinopathy is a major cause of blindness.AI models exist but  most of them have major flaws :-
they make confident mistakes or rather overconfident predictions ,most of them suffer from imbalanced datasets andlack transparency.My project solves this by using a Hybrid bayesian-transformer Modelfor uncertainty-aware DR detection.

We will use EfficientNetB0 for feature extraction,Vision Transformers for classification, Bayesian Neural Networks for uncertainty estimation,and DR-GAN++ for synthetic data generation to fix the dataset bias.
Additionally,Explainable AI features like Grad-CAM and LIME ensures clinical trust and support from doctor community.

My model first enhances fundus images through preprocessing.Then,EfficientNetB0 extracts features,and Vision-Transformers mainly Swin-Transformer classifies DR severity.Bayesian Approximation ensures reliable predictions by calculating uncertainty score.For this we will make use of Monte Carlo Dropout.To handle data imbalance ,I use DR-GAN++ for synthetic image generation. Finally,I deploy the model as a web service where doctors can upload images for analysis and my model will predict the uncertainity score and the area in the retina which make model predict the outcome.

Unlike traditional CNNs, my model combines Vision-Transformers and bayesian methods for more accurate,trustworthy predictions . It also inegrates a self-supervised learning to reduce labeled data dependencies and synthetic image generation for better training.
