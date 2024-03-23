import cv2
import os
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error

folder_path = '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part1/'

feature_vectors = []
age_labels = []

def extract_hog_features(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, feature_vector=True)
    return fd

processed_count = 0
skipped_count = 0
counter = 0
for image_file in os.listdir(folder_path):
    if image_file.endswith('.jpg'):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        counter = counter + 1
 
        hog_fd = extract_hog_features(gray)  
        if hog_fd is None:
            print(f"Yüz tespit edilemedi: {image_file} ", counter)
            skipped_count += 1
            continue

        feature_vectors.append(hog_fd)
        age_labels.append(int(image_file.split('_')[0]))
        processed_count += 1

print(f"Toplam işlenen dosya sayısı: {processed_count}")
print(f"Atlanan dosya sayısı: {skipped_count}")

feature_vectors = np.array(feature_vectors)
age_labels = np.array(age_labels)

X_train, X_test, y_train, y_test = train_test_split(feature_vectors, age_labels, test_size=0.2, random_state=42)

flann = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X_train)

predicted_ages = []
print("Tahmin işlemi başlıyor...")
for i, (test_vector, actual_age) in enumerate(zip(X_test, y_test)):
    test_vector = test_vector.reshape(1, -1)
    _, indices = flann.kneighbors(test_vector)
    predicted_age = y_train[indices[0]].mean()
    predicted_ages.append(predicted_age)
    if i % 100 == 0:
        print(f"{i+1}. örnek işleniyor... Gerçek Yaş: {actual_age}, Tahmin Edilen Yaş: {predicted_age:.2f}")

mae = mean_absolute_error(y_test, predicted_ages)
print(f"Ortalama Mutlak Hata (MAE): {mae:.2f}")
