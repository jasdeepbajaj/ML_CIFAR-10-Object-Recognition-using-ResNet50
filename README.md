# CIFAR-10 Object Recognition using ResNet50

## Description
In the domain of computer vision, accurately recognizing objects within images is a fundamental challenge. This project addresses this challenge by utilizing the CIFAR-10 dataset, a cornerstone in the field, comprising 60,000 images across 10 different classes. Our goal is to leverage deep learning techniques, specifically the ResNet50 model, to achieve high accuracy in object recognition tasks.

This initiative is part of the broader scope of Convolutional Neural Networks (CNNs) applications. By fine-tuning the ResNet50 model, pre-trained on ImageNet, for the CIFAR-10 dataset, we aim to explore the model's capabilities in understanding and categorizing images into one of the ten classes.

## Installation

To set up the project environment, follow these instructions:

```bash
# Clone the repository
git clone https://github.com/jasdeepbajaj/ML_CalorieBurntPredictionModel.git

# Navigate to the project directory
cd your-project-directory

# Install the requirements
pip install -r requirements.txt

# Run the Jupyter notebook
jupyter notebook CIFAR_10_Object_Recognition_using_ResNet50.ipynb
```
## Data

The CIFAR-10 dataset is utilized in this project, containing 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images.

## Model

The core of this project is utilizing a pre-trained ResNet50 model, adapted for the CIFAR-10 dataset. The model undergoes fine-tuning to adjust to the specific characteristics of CIFAR-10, leveraging transfer learning to significantly improve recognition accuracy.

## Features

- Data preprocessing and augmentation
- Utilization of ResNet50 for feature extraction
- Fine-tuning and training of the model on CIFAR-10
- Evaluation of model performance using accuracy and loss metrics
- Visualization of training results and model predictions

## Dependencies 

- numpy
- kaggle
- Pillow
- py7zr
- pandas
- matplotlib
- scikit-learn
- opencv-python
- tensorflow

These dependencies are also listed in the requirements.txt file.

## Documentation

For a detailed understanding of the project's methodology and implementation, refer to the comments and markdown cells within the CIFAR_10_Object_Recognition_using_ResNet50.ipynb notebook.
