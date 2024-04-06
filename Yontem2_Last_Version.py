import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random


folder_paths = [
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part1/',
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
                        raise ValueError("Görüntü okunamadı")
                    image = cv2.resize(image, img_size)  # Görüntüyü yeniden boyutlandırır
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # HOG özniteliklerini çıkarır
                    hog_feature, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, feature_vector=True)
                    hog_features.append(hog_feature)
                    age = int(image_file.split('_')[0])
                    ages.append(age)
                    print(f"İşlenen resim: {image_file} ", counter)
                    counter += 1
                    processed_count += 1
                except Exception as e:
                    print(f"{image_file} işlenirken hata oluştu: {e}")
                    skipped_count += 1
    print(f"Toplam işlenen resim sayısı: {processed_count}, İşlenemeyen resim sayısı: {skipped_count}")
    return np.array(hog_features), np.array(ages)

hog_features, ages = load_images_and_labels(folder_paths)

X_train, X_test, y_train, y_test = train_test_split(hog_features, ages, test_size=0.2, random_state=42)

# n_components değerini 28 olarak ayarlama, daha fazla yapınca hata verdi mesela 100'de
pca = PCA(n_components=28)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train_pca, y_train)

predicted_ages = knn_regressor.predict(X_test_pca)

mae = mean_absolute_error(y_test, predicted_ages)
mse = mean_squared_error(y_test, predicted_ages)
print(f"Ortalama Mutlak Hata (MAE): {mae:.2f}")
print(f"Ortalama Kare Hata (MSE): {mse:.2f}")


correct_within_5_years = sum(abs(actual - predicted) <= 5 for actual, predicted in zip(y_test, predicted_ages))
accuracy_within_5_years = correct_within_5_years / len(y_test) * 100
print(f"Doğruluk oranı (±5 yaş): {accuracy_within_5_years:.2f}%")

# Test setinden rastgele 20 örnek seçer ve görsellerle birlikte tahminleri gösterir
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
plot_sample_images(sample_image_paths, predicted_ages, y_test, 4, 5)
