import cv2
import numpy as np
import os
from datetime import datetime
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import time


start_time = time.time()


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


dataset = 'imdb'  # 'UTKFace', 'imdb', veya 'Both'


folder_paths = []
if dataset in ['UTKFace', 'Both']:
    folder_paths += [
        '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part1/',
        '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part2/',
        '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part3/'
    ]
c = 0
if dataset in ['imdb', 'Both']:
    base_path = '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/imdb_crop/'
    for subdir in os.listdir(base_path):
        if c < 1:
            c = c + 1
            folder_path = os.path.join(base_path, subdir)
            if os.path.isdir(folder_path):
                folder_paths.append(folder_path)

feature_vectors = []
age_labels = []
image_paths = []  # Görüntü yollarını saklamak için

# LBP özellikleri çıkarma fonksiyonu
def extract_lbp_features(image, P=8, R=1):
    lbp = local_binary_pattern(image, P=P, R=R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# IMDb dosya adından yaş hesaplama fonksiyonu
def calculate_age_from_imdb_filename(filename):
    parts = filename.split('_')
    dob = datetime.strptime(parts[2], "%Y-%m-%d")
    photo_year = int(parts[3].split('.')[0])
    age = photo_year - dob.year
    return age

processed_count = 0
skipped_count = 0

# Görüntü işleme ve yüz algılama
for folder_path in folder_paths:
    for image_file in os.listdir(folder_path):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Görüntü okunamadı: {image_file}")
                skipped_count += 1
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # Histogram eşitleme
            
            faces, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(32, 32), scale=1.05)
            if len(faces) == 0:
                print(f"Yüz algılanamadı: {image_file}")
                skipped_count += 1
                continue

            # Yüz bulunduysa
            for (x, y, w, h) in faces:
                cropped_face = gray[y:y+h, x:x+w]
                lbp_hist = extract_lbp_features(cropped_face)
                feature_vector = lbp_hist
                
                # Yaş hesaplama bölümü
                if dataset == 'UTKFace' or dataset == 'Both':
                    age_label = int(image_file.split('_')[0])
                else:  # 'imdb' veya 'Both', gerek kalmadı buraya
                    age_label = calculate_age_from_imdb_filename(image_file)
                
                feature_vectors.append(feature_vector)
                age_labels.append(age_label)
                image_paths.append(image_path)
                processed_count += 1

if not feature_vectors:
        print("İşlenmiş veri bulunamadı. Çıkılıyor...")
else:
    # Eğitim ve test setlerini ayırır
    X_train, X_test, y_train, y_test, X_train_paths, X_test_paths = train_test_split(
        feature_vectors, age_labels, image_paths, test_size=0.2, random_state=42
    )

    # FLANN ile en yakın komşuları hesaplama vs
    flann = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X_train)

    predicted_ages = []
    prediction_errors = []

    best_prediction_error = np.inf
    worst_prediction_error = -np.inf
    best_image_path = None
    worst_image_path = None

    
    for i, (test_vector, actual_age) in enumerate(zip(X_test, y_test)):
        test_vector = test_vector.reshape(1, -1)
        _, indices = flann.kneighbors(test_vector)
        predicted_age = y_train[indices[0]].mean()
        error = abs(predicted_age - actual_age)
        prediction_errors.append(error)

        if error < best_prediction_error:
            best_prediction_error = error
            best_image_path = X_test_paths[i]
        if error > worst_prediction_error:
            worst_prediction_error = error
            worst_image_path = X_test_paths[i]

        predicted_ages.append(predicted_age)
        if i % 100 == 0 or i == len(X_test) - 1:
            print(f"{i+1}/{len(X_test)} örnek işlendi... Gerçek Yaş: {actual_age}, Tahmin Edilen Yaş: {predicted_age:.2f}")

    mae = mean_absolute_error(y_test, predicted_ages)
    print(f"Ortalama Mutlak Hata (MAE): {mae:.2f}")

    # En iyi ve en kötü tahminlerin görsellerini gösterir
    if best_image_path and worst_image_path:
        def show_image(image_path, title):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Matplotlib için RGB formatına çevirir
            plt.figure()
            plt.imshow(image)
            plt.title(title)
            plt.axis('off')
            plt.show()

        show_image(best_image_path, f'En İyi Tahmin (Gerçek Yaş: {y_test[np.argmin(prediction_errors)]}, Tahmin Edilen Yaş: {predicted_ages[np.argmin(prediction_errors)]:.2f}, Hata: {best_prediction_error})')
        show_image(worst_image_path, f'En Kötü Tahmin (Gerçek Yaş: {y_test[np.argmax(prediction_errors)]}, Tahmin Edilen Yaş: {predicted_ages[np.argmax(prediction_errors)]:.2f}, Hata: {worst_prediction_error})')

    # Modeli kaydet, sonrası için
    joblib.dump(flann, '/home/sevvalizmirli/Desktop/Computer Vision/Github/Computer-Vision-Human-Age-Detection/flann_model.joblib')

    # Bitiş zamanı
    end_time = time.time()

    # Kodun çalışma süresini hesaplar
    total_time = end_time - start_time
    total_time_minutes = total_time / 60  # Dakikaya çevirir

    print(f"Toplam süre: {total_time:.2f} saniye ({total_time_minutes:.2f} dakika)")

