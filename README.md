# Improving the Detection of Small Pulmonary Nodules in Chest X-Rays using Learnable Downsampling Techniques

This repository contains the framework that will be used to evaluate different methodologies and techniques utilized for detecting pulmonary nodules in Chest X-Rays (CXRs).

## Project Overview:
In medical imaging, CXRs are high resolution images, training Deep Neural Network (DNN) models using this high resolution data is often very challening due to how computationally expensive the process it. Thus is it often required to downsample high resolution images so the lower resolution images can be passed through these complex models to train on. However, it has often been a recurring issue that the downsampling process loses out on valuable semantic information pertaining to smaller nodules which are required during the classification process.

Therefore this project aims to implement a modular pipeline to compare:
- **Baselines:** Standard Convolutional Neural Network (CNN) architectures with Max/Average Pooling.
- **Experimental:** CNNs integrated with learnable downsampling modules.

## Directory Structure:
```bash
IDPN_Model/
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
Clone the repository:
```bash
git clone https://github.com/SudhirAjodhya/IDPN_Model.git
```

Create virtual environment and install dependencies:
 ```bash
 python -m venv torch_env   # Create virtual environment
 source torch_env/Scripts/activate # Activating virtual environment in GitBash
 ./torch_env/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)    # install GPU-enabled PyTorch in the root directory
 ```

 ### 2. Verify Installation:
 **Local (Windows and Git bash)**
 If the environment is activated `(torch_env)`:
 ```bash
 python run_test.py # Runs the test script to verify the install
 ```

 Alternatively, use the provided shell script:
 ```bash
 ./run.sh test
 ```

 Linux:
 ```bash
 make test
 ```

### 3. Data set downloads and layout:
- JSRT [Download JSRT dataset HERE](https://www.kaggle.com/datasets/raddar/nodules-in-chest-xrays-jsrt)
Place in /data_sets, structured as follows
```python
data_sets
|--JSRT/
|  |--images/
|     |--images/
```

- NODE21 [Download Node21 dataset HERE](https://www.kaggle.com/datasets/pshikk/node-21-dataset-untampered)
Place in /data_sets, structured as follows
```python
data_sets
|--node21/
|  |--cxr_images/
|     |--original_data/
|     |--procssed_data/
```

 ### 4. Running Project:
If the environment is activated `(torch_env)`:
```bash
python <file_name>.py
``` 

Or, using the provided shell script:
```bash
./run.sh <file_name>.py
```

Or with the MakeFile, if make is installed:
```bash
make run file=<filename>.py
```
