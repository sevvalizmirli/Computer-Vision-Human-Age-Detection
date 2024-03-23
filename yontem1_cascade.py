import cv2
import os
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error

# Haar Cascade yüz tespiti için kullanışmıştır
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_lbp_features(image, P=8, R=1):
    lbp = local_binary_pattern(image, P=P, R=R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

folder_path = '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part1/'
feature_vectors = []
age_labels = []
sayac = 0
for image_file in os.listdir(folder_path):
    if image_file.endswith('.jpg'):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (500, 500))
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        sayac = sayac + 1
        if len(faces) == 0:
            print("Yüz tespit edilemedi: ", sayac, " ", image_file)
            continue
        
        for (x, y, w, h) in faces:
            cropped_face = gray[y:y+h, x:x+w]
            
            if cropped_face.size == 0:
                continue
            
            lbp_hist = extract_lbp_features(cropped_face)
            feature_vectors.append(lbp_hist)
            age_labels.append(int(image_file.split('_')[0]))

feature_vectors = np.array(feature_vectors)
age_labels = np.array(age_labels)

X_train, X_test, y_train, y_test = train_test_split(feature_vectors, age_labels, test_size=0.2, random_state=42)

flann = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X_train)

predicted_ages = np.zeros(len(X_test))
for i, test_vector in enumerate(X_test):
    test_vector = test_vector.reshape(1, -1)
    _, indices = flann.kneighbors(test_vector)
    predicted_age = y_train[indices[0]].mean()
    predicted_ages[i] = predicted_age

# Gerçek yaşlar ve tahmin edilen yaşları ekrana basar
print("\nGerçek Yaş - Tahmin Edilen Yaş")
for i in range(len(y_test)):
    print(f"{y_test[i]} - {predicted_ages[i]:.2f}")

mae = mean_absolute_error(y_test, predicted_ages)
print(f"\nOrtalama Mutlak Hata (MAE): {mae:.2f}")
