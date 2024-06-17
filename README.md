# plant-disease-DL
Plant Disease Classification using Convolutional Neural Networks (CNN)
This project aims to classify plant diseases using deep learning techniques, specifically Convolutional Neural Networks (CNNs). The dataset used is sourced from Kaggle's PlantVillage Dataset, which contains segmented color and grayscale images of various plant species affected by different diseases.

Project Overview
The project includes the following key components:

Data Acquisition and Preparation
Model Development using CNN
Training and Evaluation
Model Deployment and Prediction
Data Acquisition and Preparation
The dataset is downloaded using the Kaggle API and extracted locally. It consists of segmented images categorized into different folders based on plant species and disease type. The dataset is split into training and validation sets, and images are resized to 224x224 pixels for input into the CNN model.

Model Development using CNN
The CNN model is built using TensorFlow/Keras with the following architecture:

Layers: Two sets of Conv2D and MaxPooling2D layers for feature extraction, followed by Flatten and Dense layers for classification.
Activation Function: ReLU is used for hidden layers, and softmax is used for the output layer for multi-class classification.
Optimizer: Adam optimizer is employed for gradient descent.
Loss Function: Categorical cross-entropy is used as the loss function to measure the model's performance.
Training and Evaluation
The model is trained on the training set with data augmentation using ImageDataGenerator for better generalization. Training progress is monitored using accuracy and loss metrics, and validation accuracy is evaluated to assess the model's performance on unseen data.

Model Deployment and Prediction
Once trained, the model can predict the class of a new image of a plant leaf using the saved .h5 model file. The prediction involves preprocessing the image, making predictions using the trained model, and mapping the predicted class index back to its corresponding disease category.

Installation
To run this project, ensure you have Python installed along with the necessary libraries:

TensorFlow
NumPy
Matplotlib
Pillow (PIL)
scikit-learn
