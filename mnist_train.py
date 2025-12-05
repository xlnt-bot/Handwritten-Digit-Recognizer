import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Loading CSV Data...")

train_df = pd.read_csv("raw_data/mnist_test.csv")

print("Preprocessing data...")

y_train = train_df.iloc[:, 0].values # Just taking the values except column names
x_train = train_df.iloc[:, 1:].values

# Need to normalize the data since they are not ideal for neural network
x_train = x_train/255.0

# Reshaping the image from linear arrays to an image form
x_train = x_train.reshape(-1, 28, 28 ,1)

print(f"Data Loaded: {x_train.shape[0]} images found.")

# Building the model
model = Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dropout(.5), # Helps in learning patterns instead of memorization by turning of 50% of the neurons   
    layers.Dense(10)
])

# Compiling the model
model.compile(optimizer='adam', 
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics=['accuracy'])

# Model Summary
print("Model Summary...")
model.summary()

# Training the model
print("\nStarting Training...")
history = model.fit(x_train, y_train, epochs=5, validation_split = 0.2)

# Saving the model for reuse later
model.save('mnist_numbers.keras')
print("\nModel saved successfully as 'mnist_numbers.keras'!")

#Visualisation of accuracy and loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.show()

# Visual Prediction Check
import random
random_idx = random.randint(0, len(x_train) - 1)
sample_image = x_train[random_idx]
true_label = y_train[random_idx]

# Reshape for prediction (model expects a batch, so we add a dimension)
prediction = model.predict(sample_image.reshape(1, 28, 28, 1))
predicted_label = np.argmax(prediction)

print(f"\n--- Random Test ---")
print(f"True Label: {true_label}")
print(f"AI Predicted: {predicted_label}")

plt.imshow(sample_image.reshape(28, 28), cmap='gray')
plt.title(f"True: {true_label}, Pred: {predicted_label}")
plt.show()