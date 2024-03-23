import cv2
import dlib
import os
import numpy as np

# Dlib yükler
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Resimlerin bulunduğu klasör yolu, sonrasında bütün klasörler için güncellenecek
folder_path = '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part1/'

# Özellik vektörleri ve yaş etiketleri için boş listeler oluştur
feature_vectors = []
age_labels = []

# Klasördeki tüm resim dosyaları için, sonrasında farklı klasörler de eklenecek
for image_file in os.listdir(folder_path):
    if image_file.endswith('.jpg'):
        # Resim dosya yolu
        image_path = os.path.join(folder_path, image_file)

        # Resimi okur ve griye çevirir
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Yüzleri tespit eder
        faces = detector(gray, 1)

        for face in faces:
            # Landmark'ları bulur
            landmarks = predictor(gray, face)

            # Gözler arası mesafe ve ağız genişliği özelliklerini hesaplar
            eye_distance = np.linalg.norm(np.array([landmarks.part(36).x, landmarks.part(36).y]) - np.array([landmarks.part(45).x, landmarks.part(45).y]))
            mouth_width = np.linalg.norm(np.array([landmarks.part(48).x, landmarks.part(48).y]) - np.array([landmarks.part(54).x, landmarks.part(54).y]))

            # Özellik vektörünü ve yaş etiketini saklar
            feature_vectors.append([eye_distance, mouth_width])
            age_labels.append(int(image_file.split('_')[0]))
