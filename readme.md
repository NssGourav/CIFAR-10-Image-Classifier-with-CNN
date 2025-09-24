# CIFAR-10 Image Classifier with CNN

A beginner-friendly implementation of a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset.

## Overview

This project trains a simple CNN model to classify images into 10 categories:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

**Target Accuracy**: ≥70%

## Project Structure

```
├── main.ipynb          # Main Jupyter notebook with all code
├── models/             # Saved model files
│   └── cnn_model.h5    # Trained CNN model
├── results/            # Generated plots and visualizations
│   ├── training_history.png
│   └── sample_prediction.png
├── requirements.txt    # Python dependencies
└── readme.md          # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the notebook:
```bash
jupyter notebook main.ipynb
```

## Implementation Steps

The notebook is organized into 5 clear steps:

1. **Import Data**: Load CIFAR-10 dataset and display sample images
2. **Preprocess**: Normalize pixel values and convert labels to categorical
3. **Build CNN**: Create a simple CNN architecture with 3 convolutional blocks
4. **Train & Evaluate**: Train the model and plot accuracy/loss curves
5. **Save Model & Predict**: Save the model and demonstrate predictions

## Model Architecture

- **Input**: 32x32x3 RGB images
- **Convolutional Layers**: 3 blocks with increasing filters (32, 64, 64)
- **Pooling**: MaxPooling2D after each conv block
- **Dense Layers**: 64 neurons + dropout + 10 output neurons
- **Activation**: ReLU for hidden layers, Softmax for output

## Expected Results

- Training accuracy: ~75-80%
- Test accuracy: ~70-75%
- Model saved as HDF5 format
- Visualizations saved as PNG files

## Features

- ✅ Beginner-friendly code with clear comments
- ✅ Step-by-step implementation
- ✅ Data visualization
- ✅ Training progress plots
- ✅ Sample prediction with confidence scores
- ✅ Model persistence
- ✅ Results organization
