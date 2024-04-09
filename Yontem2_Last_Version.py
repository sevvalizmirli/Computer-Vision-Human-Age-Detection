import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

folder_paths = ['/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part1/',
                '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part2/',
                '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part3/'
                
                ]

def load_images_and_labels(paths, img_size=(128, 128)):
    counter = 0
    hog_features = []
    ages = []
    processed_count = 0
    skipped_count = 0
    for folder_path in paths:
        for image_file in os.listdir(folder_path):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(folder_path, image_file)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError("Image could not be read")
                    image = cv2.resize(image, img_size)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    hog_feature, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, feature_vector=True)
                    hog_features.append(hog_feature)
                    age = int(image_file.split('_')[0])
                    ages.append(age)
                    print(f"İşlenen resim: {image_file} ", counter)
                    counter += 1
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    skipped_count += 1
    print(f"Processed images: {processed_count}, Skipped images: {skipped_count}")
    return np.array(hog_features), np.array(ages)

hog_features, ages = load_images_and_labels(folder_paths)
X_train_val, X_test, y_train_val, y_test = train_test_split(hog_features, ages, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

pca = PCA(n_components=28)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

# Scaling features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_pca)
X_val_scaled = scaler.transform(X_val_pca)
X_test_scaled = scaler.transform(X_test_pca)

knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train_pca, y_train)

svm_regressor = SVR(C=1.0, epsilon=0.2)
svm_regressor.fit(X_train_scaled, y_train)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_pca, y_train)

# Predictions
predicted_ages_val_knn = knn_regressor.predict(X_val_pca)
predicted_ages_test_knn = knn_regressor.predict(X_test_pca)

predicted_ages_val_svm = svm_regressor.predict(X_val_scaled)
predicted_ages_test_svm = svm_regressor.predict(X_test_scaled)

predicted_ages_val_rf = rf_regressor.predict(X_val_pca)
predicted_ages_test_rf = rf_regressor.predict(X_test_pca)

def custom_error_metric(y_true, y_pred):
    error = 0
    for true, pred in zip(y_true, y_pred):
        if abs(true - pred) > 5:
            error += abs(true - pred)
    return error / len(y_true)

# Custom MAE calculations
custom_mae_val_knn = custom_error_metric(y_val, predicted_ages_val_knn)
custom_mae_test_knn = custom_error_metric(y_test, predicted_ages_test_knn)

custom_mae_val_svm = custom_error_metric(y_val, predicted_ages_val_svm)
custom_mae_test_svm = custom_error_metric(y_test, predicted_ages_test_svm)

custom_mae_val_rf = custom_error_metric(y_val, predicted_ages_val_rf)
custom_mae_test_rf = custom_error_metric(y_test, predicted_ages_test_rf)

print(f"KNN Custom Validation MAE: {custom_mae_val_knn:.2f}")
print(f"KNN Custom Test MAE: {custom_mae_test_knn:.2f}")

print(f"SVM Custom Validation MAE: {custom_mae_val_svm:.2f}")
print(f"SVM Custom Test MAE: {custom_mae_test_svm:.2f}")

print(f"RF Custom Validation MAE: {custom_mae_val_rf:.2f}")
print(f"RF Custom Test MAE: {custom_mae_test_rf:.2f}")

def calculate_accuracy_within_5_years(y_true, y_pred):
    correct_within_5_years = sum(abs(actual - predicted) <= 5 for actual, predicted in zip(y_true, y_pred))
    accuracy_within_5_years = correct_within_5_years / len(y_true) * 100
    return accuracy_within_5_years

accuracy_within_5_years_knn = calculate_accuracy_within_5_years(y_test, predicted_ages_test_knn)
accuracy_within_5_years_svm = calculate_accuracy_within_5_years(y_test, predicted_ages_test_svm)
accuracy_within_5_years_rf = calculate_accuracy_within_5_years(y_test, predicted_ages_test_rf)

print(f"KNN Accuracy within ±5 years: {accuracy_within_5_years_knn:.2f}%")
print(f"SVM Accuracy within ±5 years: {accuracy_within_5_years_svm:.2f}%")
print(f"RF Accuracy within ±5 years: {accuracy_within_5_years_rf:.2f}%")

def plot_sample_images(image_paths, predicted_ages, actual_ages, rows=4, cols=5):
    fig = plt.figure(figsize=(20, 10))
    sampled_indices = random.sample(range(len(image_paths)), 20)
    for i, idx in enumerate(sampled_indices):
        img = cv2.imread(image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.title(f"Pred: {predicted_ages[idx]:.1f}, Actual: {actual_ages[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

sample_image_paths = [folder_paths[0] + os.listdir(folder_paths[0])[i] for i in random.sample(range(len(y_test)), len(y_test))]
plot_sample_images(sample_image_paths, predicted_ages_test_knn, y_test, 4, 5)
