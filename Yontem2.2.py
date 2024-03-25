import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

folder_paths = [
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part4/'
]

def preprocess_and_extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (128, 128))
    image = cv2.medianBlur(image, 5)
    image = cv2.equalizeHist(image)
    fd, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1), visualize=True, feature_vector=True)
    return fd

feature_vectors, age_labels, image_paths = [], [], []

for folder_path in folder_paths:
    for image_file in os.listdir(folder_path):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(folder_path, image_file)
            fd = preprocess_and_extract_features(image_path)
            if fd is not None:
                feature_vectors.append(fd)
                age_label = int(image_file.split('_')[0])
                age_labels.append(age_label)
                image_paths.append(image_path)

X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    feature_vectors, age_labels, image_paths, test_size=0.2, random_state=42
)

model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, verbose=True)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Accuracy: {accuracy}")

cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, predictions))

errors = np.abs(y_test - predictions)
best_pred_idx = np.argmin(errors)
worst_pred_idx = np.argmax(errors)

def show_image(image_path, title):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

show_image(paths_test[best_pred_idx], f'Best Prediction\nTrue Age: {y_test[best_pred_idx]}\nPredicted Age: {predictions[best_pred_idx]}')
show_image(paths_test[worst_pred_idx], f'Worst Prediction\nTrue Age: {y_test[worst_pred_idx]}\nPredicted Age: {predictions[worst_pred_idx]}')
