import cv2
import dlib
import os
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

folder_paths = [
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part1/',
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part2/',
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part3/'
]

feature_vectors = []
age_labels = []

def extract_lbp_features(image, P=8, R=1):
    lbp = local_binary_pattern(image, P=P, R=R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

processed_count = 0
skipped_count = 0
counter = 0

for folder_path in folder_paths:  # Her bir klasör için
    for image_file in os.listdir(folder_path):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 1)
            counter = counter+1
            if len(faces) == 0:
                print(f"Yüz tespit edilemedi: {image_file} ", counter)
                skipped_count += 1
                continue

            for face in faces:
                landmarks = predictor(gray, face)
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y
                if w <= 0 or h <= 0:
                    print(f"Boş yüz bölgesi: {image_file}")
                    skipped_count += 1
                    continue

                cropped_face = gray[y:y+h, x:x+w]
                if cropped_face.size == 0:
                    print(f"Kırpılmış yüz boş: {image_file}")
                    skipped_count += 1
                    continue

                lbp_hist = extract_lbp_features(cropped_face)
                eye_distance = np.linalg.norm(np.array([landmarks.part(36).x, landmarks.part(36).y]) - np.array([landmarks.part(45).x, landmarks.part(45).y]))
                mouth_width = np.linalg.norm(np.array([landmarks.part(48).x, landmarks.part(48).y]) - np.array([landmarks.part(54).x, landmarks.part(54).y]))
                feature_vector = np.concatenate(([eye_distance, mouth_width], lbp_hist))
                feature_vectors.append(feature_vector)
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
    if i % 100 == 0 or i == len(X_test) - 1:  # İlerleme durumunu yazdırır
                # Her 100 örnekte bir veya son örnekte ilerleme durumunu yazdırır
        print(f"{i+1}/{len(X_test)} örnek işleniyor... Gerçek Yaş: {actual_age}, Tahmin Edilen Yaş: {predicted_age:.2f}")


mae = mean_absolute_error(y_test, predicted_ages)
print(f"Ortalama Mutlak Hata (MAE): {mae:.2f}")


detailed_results = zip(y_test, predicted_ages)
for index, (actual, predicted) in enumerate(detailed_results):
    if index < 20:  
        print(f"Test Örneği {index + 1} - Gerçek Yaş: {actual}, Tahmin Edilen Yaş: {predicted:.2f}")
