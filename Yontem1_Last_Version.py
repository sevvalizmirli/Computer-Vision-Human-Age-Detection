import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import joblib

folder_paths = [
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part1/',
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part2/',
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part3/'
]

def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray
    except Exception as e:
        print(f"Resim okunamadı: {image_path}, Hata: {e}")
        return None

def extract_lbp_features(face_image, P=8, R=1):
    lbp = local_binary_pattern(face_image, P=P, R=R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

feature_vectors = []
age_labels = []
processed_count = 0
skipped_count = 0
image_paths = []  
counter = 0
for folder_path in folder_paths:
    for image_file in os.listdir(folder_path):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(folder_path, image_file)
            print(f"İşlenen resim: {image_file} ", counter)
            counter = counter + 1
            image = preprocess_image(image_path)
            if image is None:
                skipped_count += 1
                continue
            lbp_features = extract_lbp_features(image)
            feature_vectors.append(lbp_features)
            age_label = int(image_file.split('_')[0])
            age_labels.append(age_label)
            image_paths.append(image_path)  
            processed_count += 1

print(f"Toplam değerlendirilen resim sayısı: {processed_count}")
print(f"Toplam atlanan resim sayısı: {skipped_count}")

feature_vectors = np.array(feature_vectors)
age_labels = np.array(age_labels)
image_paths = np.array(image_paths)  

# Veriyi eğitim, doğrulama ve test setlerine ayırma
X_train, X_test, y_train, y_test, train_paths, test_paths = train_test_split(feature_vectors, age_labels, image_paths, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test, val_paths, test_paths = train_test_split(X_test, y_test, test_paths, test_size=0.5, random_state=42)

flann = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X_train)
predicted_ages = [y_train[flann.kneighbors([x], return_distance=False)[0]].mean() for x in X_test]

joblib.dump(flann, '/home/sevvalizmirli/Desktop/Computer Vision/Github/Computer-Vision-Human-Age-Detection/flann_model.joblib')

correct_within_5_years = sum(abs(a - p) <= 5 for a, p in zip(y_test, predicted_ages))
accuracy_within_5_years = correct_within_5_years / len(y_test) * 100
print(f"Doğruluk oranı (±5 yaş): {accuracy_within_5_years:.2f}%")

updated_mae = np.mean([abs(a - p) if abs(a - p) <= 5 else 0 for a, p in zip(y_test, predicted_ages)])
updated_mse = np.mean([(a - p) ** 2 if abs(a - p) <= 5 else 0 for a, p in zip(y_test, predicted_ages)])
print(f"Ortalama Mutlak Hata (MAE): {updated_mae:.2f}")
print(f"Ortalama Kare Hata (MSE): {updated_mse:.2f}")

def plot_sample_images(image_paths, predicted_ages, actual_ages, rows=4, cols=5):
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
    for ax, img_path, pred, actual in zip(axs.flat, image_paths, predicted_ages, actual_ages):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f"Pred: {pred:.1f}\nActual: {actual}")
        ax.axis('off')

sample_indexes = random.sample(range(len(test_paths)), 20)
sample_image_paths = test_paths[sample_indexes]
sample_predicted_ages = [predicted_ages[i] for i in sample_indexes]
sample_actual_ages = [y_test[i] for i in sample_indexes]

plot_sample_images(sample_image_paths, sample_predicted_ages, sample_actual_ages)
plt.tight_layout()
plt.show()
