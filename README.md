# Monkeypox Detection Using Binary Image Classification

This repository contains code and dataset for generating a model to detect monkeypox from user-inputted video images using the ResNet50 architecture. The project focuses on binary class image classification, specifically classifying skin lesion images into different categories related to monkeypox. The repository includes the necessary code and steps to train and evaluate the model.

## Dataset

The dataset used in this project can be downloaded from [this link](https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset?select=Monkeypox_Dataset_metadata.csv). It includes a collection of skin lesion images related to monkeypox along with metadata.

## Repository Structure and Code

The repository contains the following code files:

- `import os`: Importing the necessary libraries.
- `import cv2`: Importing OpenCV for image processing.
- `import numpy as np`: Importing NumPy for array manipulation.
- `from sklearn.model_selection import train_test_split`: Splitting the dataset into training and validation sets.
- `from keras.preprocessing.image import ImageDataGenerator`: Creating data generators for image augmentation.
- `from keras.models import Sequential`: Creating a sequential model.
- `from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization`: Adding layers to the model.
- `from keras.callbacks import EarlyStopping, ModelCheckpoint`: Implementing early stopping and model checkpointing.
- `from tensorflow.keras.applications import ResNet50V2`: Importing the ResNet50V2 pre-trained model.

The code also includes the steps for data loading, defining the model architecture, compiling the model, fitting the model to the data, and saving the trained model.

## Usage

To use this repository and detect monkeypox from video images, follow these steps:

1. Download the dataset from the provided link and place it in a folder named `Dataset`.

2. Run the code provided in the repository. Ensure that you have the necessary dependencies installed.

3. The code will preprocess the data, create a ResNet50-based model, train the model, and save it as `monkeypox_model.h5`.

4. Once the model is trained and saved, you can use it to classify new video images for monkeypox detection.

Feel free to explore the code and modify it according to your requirements. You can experiment with different model architectures, hyperparameters, or techniques to improve the classification accuracy.

## Contributing

Contributions to this repository are highly welcome. If you have any ideas, suggestions, or improvements, please feel free to fork the repository and submit a pull request. Let's collaborate to enhance the monkeypox detection model!

## License

This project is licensed under the [MIT License](LICENSE). You are free to use and modify the code for your own purposes.
