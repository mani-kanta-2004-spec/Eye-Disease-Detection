import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Image size (must match training dimensions)
IMG_HEIGHT = 512
IMG_WIDTH = 512

# Custom objects dictionary for potential compatibility fixes
custom_objects = {
    'InputLayer': tf.keras.layers.InputLayer
}

# Function to load and preprocess a single image
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image at {image_path}")
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_normalized = img / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=0)
    return img, img_normalized

# Function to display results
def display_results(original_img, predicted_mask):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Segmentation Mask')
    plt.axis('off')
    plt.show()

# Main prediction function
def predict_mask(model, image_path):
    original_img, preprocessed_img = load_and_preprocess_image(image_path)
    predicted_mask = model.predict(preprocessed_img)
    predicted_mask = predicted_mask[0]
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
    return original_img, predicted_mask

# Load the trained model
model_path = r"D:\sem6\Computer vision and Image processing\Project\Datasets\retina_segmentation_unet.h5"
try:
    model = load_model(model_path, custom_objects=custom_objects, compile=False)
    print("Model loaded successfully!")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Test the model with an image
test_image_path = r"D:\sem6\Computer vision and Image processing\Project\all\dataset\00. Normal\img_0001.jpg" # Update this to your actual image path

try:
    original_img, predicted_mask = predict_mask(model, test_image_path)
    display_results(original_img, predicted_mask)
    mask_output_path = "predicted_mask.png"
    cv2.imwrite(mask_output_path, predicted_mask * 255)
    print(f"Predicted mask saved to {mask_output_path}")
except Exception as e:
    print(f"Error during prediction: {e}")