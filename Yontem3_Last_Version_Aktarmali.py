import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

folder_paths = [
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part1/',
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part2/',
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part3/'
]

def load_and_preprocess_images(folder_paths, img_size=(160, 160)):
    images = []
    ages = []
    for folder_path in folder_paths:
        for image_file in os.listdir(folder_path):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(folder_path, image_file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, img_size)
                images.append(image)
                age = int(image_file.split('_')[0])
                ages.append(age)
    images = np.array(images, dtype='float32')
    ages = np.array(ages)
    images /= 255.0  # Normalize images to the range [0, 1]
    return images, ages

images, ages = load_and_preprocess_images(folder_paths)
X_train, X_temp, y_train, y_temp = train_test_split(images, ages, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Load the MobileNetV2 model with imagenet weights, excluding its top dense layer
base_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the convolutional base

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # A dense layer with 1024 units
predictions = Dense(1)(x)  # Final output layer with 1 unit for age prediction

# Compile the model
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

plt.plot(history.history['mae'], label='MAE (training data)')
plt.plot(history.history['val_mae'], label='MAE (validation data)')
plt.title('MAE for Age Prediction')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

# Evaluate the model on test data
test_predictions = model.predict(X_test).flatten()

def custom_error_metric(y_true, y_pred):
    error = np.sum(np.abs(y_true - y_pred))
    return error / len(y_true)

custom_mae = custom_error_metric(y_test, test_predictions)
custom_mse = np.mean((y_test - test_predictions) ** 2)
accuracy = np.mean(np.abs(y_test - test_predictions) <= 5) * 100

print(f"Custom Test MAE: {custom_mae:.2f}")
print(f"Custom Test MSE: {custom_mse:.2f}")
print(f"Accuracy (±5 years): {accuracy:.2f}%")

def plot_sample_predictions(images, y_true, y_pred, sample_size=20):
    indices = np.random.choice(range(len(images)), size=sample_size, replace=False)
    fig, axes = plt.subplots(4, 5, figsize=(20, 10))
    axes = axes.ravel()
    for i, idx in enumerate(indices):
        img, true_age, pred_age = images[idx], y_true[idx], y_pred[idx]
        axes[i].imshow(img)
        axes[i].set_title(f"True: {true_age}, Pred: {np.round(pred_age, 1)}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

plot_sample_predictions(X_test, y_test, test_predictions, sample_size=20)
