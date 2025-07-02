import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, concatenate
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set dataset paths
dataset_path = "Unified_Retina_Dataset"
image_dir = os.path.join(dataset_path, "images")
mask_dir = os.path.join(dataset_path, "masks")

# Image size
IMG_HEIGHT = 512
IMG_WIDTH = 512
BATCH_SIZE = 8
EPOCHS = 20

# Load images and masks
def load_data(image_dir, mask_dir):
    images = []
    masks = []
    
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0  # Normalize

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        mask = mask / 255.0  # Normalize to [0,1]
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Load dataset
images, masks = load_data(image_dir, mask_dir)
print(f"Loaded {len(images)} images and {len(masks)} masks.")

# Split dataset into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# U-Net Model
def build_unet():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Encoder (Contracting Path)
    c1 = Conv2D(64, (3,3), activation="relu", padding="same")(inputs)
    c1 = Conv2D(64, (3,3), activation="relu", padding="same")(c1)
    p1 = MaxPooling2D((2,2))(c1)

    c2 = Conv2D(128, (3,3), activation="relu", padding="same")(p1)
    c2 = Conv2D(128, (3,3), activation="relu", padding="same")(c2)
    p2 = MaxPooling2D((2,2))(c2)

    c3 = Conv2D(256, (3,3), activation="relu", padding="same")(p2)
    c3 = Conv2D(256, (3,3), activation="relu", padding="same")(c3)
    p3 = MaxPooling2D((2,2))(c3)

    c4 = Conv2D(512, (3,3), activation="relu", padding="same")(p3)
    c4 = Conv2D(512, (3,3), activation="relu", padding="same")(c4)
    p4 = MaxPooling2D((2,2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3,3), activation="relu", padding="same")(p4)
    c5 = Conv2D(1024, (3,3), activation="relu", padding="same")(c5)

    # Decoder (Expanding Path)
    u6 = UpSampling2D((2,2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3,3), activation="relu", padding="same")(u6)
    c6 = Conv2D(512, (3,3), activation="relu", padding="same")(c6)

    u7 = UpSampling2D((2,2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3,3), activation="relu", padding="same")(u7)
    c7 = Conv2D(256, (3,3), activation="relu", padding="same")(c7)

    u8 = UpSampling2D((2,2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3,3), activation="relu", padding="same")(u8)
    c8 = Conv2D(128, (3,3), activation="relu", padding="same")(c8)

    u9 = UpSampling2D((2,2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3,3), activation="relu", padding="same")(u9)
    c9 = Conv2D(64, (3,3), activation="relu", padding="same")(c9)

    outputs = Conv2D(1, (1,1), activation="sigmoid")(c9)

    model = Model(inputs, outputs)
    return model

# Compile and train model
model = build_unet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, Y_train, 
                    validation_data=(X_val, Y_val), 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS)

# Save model
model.save("retina_segmentation_unet.h5")

# Plot training results
plt.figure(figsize=(12,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()
