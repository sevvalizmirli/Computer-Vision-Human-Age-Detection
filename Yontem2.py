import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

folder_paths = ['/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part3/']

def preprocess_and_extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (128, 128))
    image = cv2.medianBlur(image, 5)
    image = cv2.equalizeHist(image)
    fd, _ = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    return fd

feature_vectors = []
age_labels = []
image_paths = []

for folder_path in folder_paths:
    for image_file in os.listdir(folder_path):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(folder_path, image_file)
            features = preprocess_and_extract_features(image_path)
            if features is not None:
                feature_vectors.append(features)
                age_labels.append(int(image_file.split('_')[0]))
                image_paths.append(image_path)

X_train, X_test, y_train, y_test, X_train_paths, X_test_paths = train_test_split(
    feature_vectors, age_labels, image_paths, test_size=0.2, random_state=42
)

model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, verbose=True, random_state=1, learning_rate_init=.001)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")

# Hataları ve en iyi/kötü tahminleri bul
errors = np.abs(predictions - y_test)
best_prediction_idx = np.argmin(errors)
worst_prediction_idx = np.argmax(errors)

def show_image(image_path, title):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

show_image(X_test_paths[best_prediction_idx], f'En İyi Tahmin\nGerçek Yaş: {y_test[best_prediction_idx]}\nTahmin Edilen Yaş: {predictions[best_prediction_idx]}')
show_image(X_test_paths[worst_prediction_idx], f'En Kötü Tahmin\nGerçek Yaş: {y_test[worst_prediction_idx]}\nTahmin Edilen Yaş: {predictions[worst_prediction_idx]}')
