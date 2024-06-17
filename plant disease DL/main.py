# Install necessary library
!pip install kaggle

# Import required libraries
import random
import numpy as np
import tensorflow as tf
import os
import json
from zipfile import ZipFile
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Ensure reproducibility by setting seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load Kaggle API credentials from JSON file
kaggle_credentials = json.load(open("kaggle.json"))
os.environ['KAGGLE_USERNAME'] = kaggle_credentials["username"]
os.environ['KAGGLE_KEY'] = kaggle_credentials["key"]

# Download the dataset using Kaggle API
!kaggle datasets download -d abdallahalidev/plantvillage-dataset

# Extract the dataset
with ZipFile("plantvillage-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall()

# List the contents of the extracted dataset
print(os.listdir("plantvillage dataset"))

# Check the number of files in different folders
print(f"Segmented images count: {len(os.listdir('plantvillage dataset/segmented'))}")
print(f"Sample segmented images: {os.listdir('plantvillage dataset/segmented')[:5]}")
print(f"Color images count: {len(os.listdir('plantvillage dataset/color'))}")
print(f"Sample color images: {os.listdir('plantvillage dataset/color')[:5]}")
print(f"Grayscale images count: {len(os.listdir('plantvillage dataset/grayscale'))}")
print(f"Sample grayscale images: {os.listdir('plantvillage dataset/grayscale')[:5]}")
print(f"Grape healthy images count: {len(os.listdir('plantvillage dataset/color/Grape___healthy'))}")
print(f"Sample grape healthy images: {os.listdir('plantvillage dataset/color/Grape___healthy')[:5]}")

# Define base directory for the dataset
base_dir = 'plantvillage dataset/color'

# Display a sample image
sample_image_path = os.path.join(base_dir, 'Apple___Cedar_apple_rust/025b2b9a-0ec4-4132-96ac-7f2832d0db4a___FREC_C.Rust 3655.JPG')
sample_img = mpimg.imread(sample_image_path)
plt.imshow(sample_img)
plt.axis('off')
plt.show()

# Parameters for image processing
image_size = 224
batch_size = 32

# Image data augmentation and normalization
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Create training data generator
train_gen = data_gen.flow_from_directory(
    base_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)

# Create validation data generator
val_gen = data_gen.flow_from_directory(
    base_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
)

# Build the CNN model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

# Display model architecture
cnn_model.summary()

# Compile the model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = cnn_model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // batch_size,
    epochs=5,
    validation_data=val_gen,
    validation_steps=val_gen.samples // batch_size
)

# Evaluate the model
print("Evaluating model performance...")
val_loss, val_accuracy = cnn_model.evaluate(val_gen, steps=val_gen.samples // batch_size)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# Function to preprocess images for prediction
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to predict image class
def predict_image(model, image_path, class_indices):
    processed_img = preprocess_image(image_path)
    predictions = model.predict(processed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name

# Map class indices to class names
class_indices = {v: k for k, v in train_gen.class_indices.items()}
json.dump(class_indices, open('class_indices.json', 'w'))

# Predict class for a new image
test_image_path = '/content/test_apple_black_rot.JPG'
predicted_class = predict_image(cnn_model, test_image_path, class_indices)
print("Predicted Class Name:", predicted_class)

# Save the model
cnn_model.save('plant_disease_prediction_model.h5')
