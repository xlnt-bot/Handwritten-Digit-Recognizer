import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd

#Loading the saved model
print("Loading saved model...")
try:
    model = keras.models.load_model('mnist_numbers.keras')
    print("Model loaded successfully!")
except:
    print("Error: Could not find 'my_mnist_model.keras'. Please run the training script first.")
    exit()

print("Loading test data...")
try:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
except FileNotFoundError:
    print("Error: Could not find 'raw_data/mnist_test.csv'.")
    exit()


x_test = x_test/255.0

x_test = x_test.reshape(-1, 28, 28, 1)

print(f"Data Loaded: {x_test.shape[0]} images found.")

print("\nEvaluationg model performance...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2)

print(f"\n--------------------------------")
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
print(f"--------------------------------")

# 5. Visual Sanity Check
# Let's check 5 random images to see where it succeeds/fails
num_samples = 5
indices = np.random.choice(len(x_test), num_samples, replace=False)

plt.figure(figsize=(15, 3))

for i, idx in enumerate(indices):
    image = x_test[idx]
    true_label = y_test[idx]
    
    # Predict
    # Add batch dimension (1, 28, 28, 1)
    prediction = model.predict(image.reshape(1, 28, 28, 1), verbose=0)
    predicted_label = np.argmax(prediction)
    
    # Plot
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(image.reshape(28, 28), cmap='gray')
    
    # Color title green if correct, red if wrong
    color = 'green' if predicted_label == true_label else 'red'
    plt.title(f"True: {true_label}\nPred: {predicted_label}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()