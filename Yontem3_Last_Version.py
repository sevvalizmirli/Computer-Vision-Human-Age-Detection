import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Eksik olan import
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Veri seti yolları
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
X_train, X_test, y_train, y_test = train_test_split(images, ages, test_size=0.2, random_state=42)

model = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Modelin eğitim ve doğrulama verisi üzerinde MAE değerlerini çizdirme
plt.plot(history.history['mae'], label='MAE (training data)')
plt.plot(history.history['val_mae'], label='MAE (validation data)')
plt.title('MAE for Age Prediction')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

# Test seti üzerinde tahminlerin hesaplanması
test_predictions = model.predict(X_test).flatten()

# Test seti üzerindeki MAE ve MSE'nin hesaplanması
mae = mean_absolute_error(y_test, test_predictions)
mse = mean_squared_error(y_test, test_predictions)

# Doğruluk oranını hesaplama
correct_predictions = np.abs(y_test - test_predictions) <= 5
accuracy = np.mean(correct_predictions) * 100

print(f"Test MAE: {mae:.2f}")
print(f"Test MSE: {mse:.2f}")
print(f"Accuracy (±5 years): {accuracy:.2f}%")

# Test setinden rastgele 20 örnek seç ve görselleri ile birlikte tahminleri göster
def plot_sample_predictions(images, y_true, y_pred, sample_size=20):
    indices = np.random.choice(range(len(images)), size=sample_size, replace=False)
    fig, axes = plt.subplots(4, 5, figsize=(20, 10))
    axes = axes.ravel()

    for i, idx in enumerate(indices):
        img, true_age, pred_age = images[idx], y_true[idx], y_pred[idx]
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"True: {true_age}, Pred: {np.round(pred_age, 1)}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

plot_sample_predictions(X_test, y_test, test_predictions, sample_size=20)
