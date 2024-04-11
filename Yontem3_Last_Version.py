# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# from sklearn.model_selection import train_test_split
# import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt

# folder_paths = [
#     '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part1/'
# ]

# def load_and_preprocess_images(folder_paths, img_size=(64, 64)):
#     images = []
#     ages = []
#     for folder_path in folder_paths:
#         for image_file in os.listdir(folder_path):
#             if image_file.endswith('.jpg'):
#                 image_path = os.path.join(folder_path, image_file)
#                 image = cv2.imread(image_path)
#                 image = cv2.resize(image, img_size)
#                 images.append(image)
#                 age = int(image_file.split('_')[0])
#                 ages.append(age)
#     images = np.array(images)
#     ages = np.array(ages)
#     return images, ages

# images, ages = load_and_preprocess_images(folder_paths)
# X_train, X_test, y_train, y_test = train_test_split(images, ages, test_size=0.2, random_state=42)

# model = Sequential([
#     Input(shape=(64, 64, 3)),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(1)
# ])

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# # Learning rate reduction
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# # Early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[reduce_lr, early_stopping])

# plt.plot(history.history['mae'], label='MAE (training data)')
# plt.plot(history.history['val_mae'], label='MAE (validation data)')
# plt.title('MAE for Age Prediction')
# plt.ylabel('MAE value')
# plt.xlabel('No. epoch')
# plt.legend(loc="upper left")
# plt.show()

# test_predictions = model.predict(X_test).flatten()

# def custom_error_metric(y_true, y_pred):
#     error = 0
#     correct = 0
#     for true, pred in zip(y_true, y_pred):
#         if abs(true - pred) > 5:
#             error += abs(true - pred)
#         else:
#             correct += 1
#     return error / len(y_true), correct / len(y_true) * 100

# custom_mae, accuracy = custom_error_metric(y_test, test_predictions)

# print(f"Custom Test MAE: {custom_mae:.2f}")
# print(f"Accuracy (±5 years): {accuracy:.2f}%")

# def plot_sample_predictions(images, y_true, y_pred, sample_size=20):
#     indices = np.random.choice(range(len(images)), size=sample_size, replace=False)
#     fig, axes = plt.subplots(4, 5, figsize=(20, 10))
#     axes = axes.ravel()
#     for i, idx in enumerate(indices):
#         img, true_age, pred_age = images[idx], y_true[idx], y_pred[idx]
#         axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         axes[i].set_title(f"True: {true_age}, Pred: {np.round(pred_age, 1)}")
#         axes[i].axis('off')
#     plt.tight_layout()
#     plt.show()

# plot_sample_predictions(X_test, y_test, test_predictions, sample_size=20)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

folder_paths = [
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part1/',
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part2/',
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part3/'
]

def load_and_preprocess_images(folder_paths, img_size=(64, 64)):
    images = []
    ages = []
    for folder_path in folder_paths:
        for image_file in os.listdir(folder_path):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(folder_path, image_file)
                image = cv2.imread(image_path)
                image = cv2.resize(image, img_size)
                images.append(image)
                age = int(image_file.split('_')[0])
                ages.append(age)
    images = np.array(images)
    ages = np.array(ages)
    return images, ages

images, ages = load_and_preprocess_images(folder_paths)
X_train, X_temp, y_train, y_temp = train_test_split(images, ages, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Veri artırma
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

model = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Learning rate reduction and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# Veri artırma ile eğitim
train_generator = data_gen.flow(X_train, y_train, batch_size=32)
history = model.fit(train_generator, epochs=30, validation_data=(X_val, y_val), callbacks=[reduce_lr, early_stopping])

plt.plot(history.history['mae'], label='MAE (training data)')
plt.plot(history.history['val_mae'], label='MAE (validation data)')
plt.title('MAE for Age Prediction')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

test_predictions = model.predict(X_test).flatten()

def custom_error_metric(y_true, y_pred, threshold=5):
    correct = np.abs(y_true - y_pred) <= threshold
    custom_mae = np.mean(np.abs(y_true[correct] - y_pred[correct]))
    custom_mse = np.mean((y_true[correct] - y_pred[correct]) ** 2)
    accuracy = np.mean(correct) * 100
    return custom_mae, custom_mse, accuracy

custom_mae, custom_mse, accuracy = custom_error_metric(y_test, test_predictions)

print(f"Custom Test MAE: {custom_mae:.2f}")
print(f"Custom Test MSE: {custom_mse:.2f}")
print(f"Accuracy (±5 years): {accuracy:.2f}%")

def plot_sample_predictions(images, y_true, y_pred, sample_size=20):
    indices = np.random.choice(range(len(images)), size=sample_size, replace=False)
    fig, axes = plt.subplots(4, 5, figsize=(20, 10))
    axes = axes.ravel()
    for i, idx in enumerate(indices):
        img, true_age, pred_age = images[idx], y_true[idx], y_pred[idx]
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Real Age: {true_age}, Pred: {np.round(pred_age, 1)}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

plot_sample_predictions(X_test, y_test, test_predictions, sample_size=20)
