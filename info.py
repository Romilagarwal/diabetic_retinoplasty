# Core Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# PyTorch & Deep Learning
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout

# Computer Vision
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Bayesian Deep Learning
import pyro
import pyro.distributions as dist

# Vision Transformers
from transformers import SwinForImageClassification

# Deployment
from flask import Flask, request, jsonify
from fastapi import FastAPI
import uvicorn

# Check PyTorch & TensorFlow versions
print("PyTorch Version: ", torch.__version__)
print("TensorFlow Version: ", tf.__version__)
