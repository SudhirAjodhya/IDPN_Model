# Improving the Detection of Small Pulmonary Nodules in Chest X-Rays using Learnable Downsampling Techniques

This repository contains the framework that will be used to evaluate different methodologies and techniques utilized for detecting pulmonary nodules in Chest X-Rays (CXRs).

## Project Overview:
In medical imaging, CXRs are high resolution images, training Deep Neural Network (DNN) models using this high resolution data is often very challening due to how computationally expensive the process it. Thus is it often required to downsample high resolution images so the lower resolution images can be passed through these complex models to train on. However, it has often been a recurring issue that the downsampling process loses out on valuable semantic information pertaining to smaller nodules which are required during the classification process.

Therefore this project aims to implement a modular pipeline to compare:
- **Baselines:** Standard Convolutional Neural Network (CNN) architectures with Max/Average Pooling.
- **Experimental:** CNNs integrated with learnable downsampling modules.

## Directory Structure:
```bash
Model/
|--- data_sets/            # Various CXR data sets should go here
|   |-- raw/               # Original DICOM png images
|   |-- processed/         # Potentially pre-processed images (resized, normalized, and split (train/val/test))
|--- src/                  # Core logic and functionality
|   |-- models/            
|   |   |-- classifiers/   # Backbone network classification models (CNN, ResNet, ImageNet, EfficientDet)
|   |   |-- downsamplers/  # Static and learnable downsamplers
|   |-- data/              # Dataset classes and augmentation lgic
|   |-- visualization/     # Nodule Visualization (Heatmaps, circled nodules etc.)
|   |-- utils/             # Helper funtions (loggin, metrics etc.)
|-- scripts/               # Entry point for execution
|   |-- train.py           # Main training and evaluation script
|-- outputs/               # Auto-generated results
|   |-- models/            # Saved weights (.pth)
|   |-- results/           # Graphs (AUC/ROC) and CSV logs
|-- requirements.txt       # Project dependencies
```

## Getting Started:
### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/SudhirAjodhya/IDPN_Model.git
cd Model
pip install requirements and dependencies
```