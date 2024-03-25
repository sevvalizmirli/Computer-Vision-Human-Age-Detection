import cv2
import dlib
import numpy as np
import os
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error
import joblib
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Veri seti seçimi
dataset = 'Both'  # 'UTKFace', 'imdb', veya 'Both'

folder_paths = []
if dataset in ['UTKFace', 'Both']:
    folder_paths += [
        '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part1/',
        '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part2/',
        '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part3/'
    ]

if dataset in ['imdb', 'Both']:
    base_path = '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/imdb_0/imdb/'
    for subdir in os.listdir(base_path):
        folder_path = os.path.join(base_path, subdir)
        if os.path.isdir(folder_path):
            folder_paths.append(folder_path)

feature_vectors = []
age_labels = []
image_paths = []

def extract_lbp_features(image, P=8, R=1):
    lbp = local_binary_pattern(image, P=P, R=R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def calculate_age_from_imdb_filename(filename):
    # IMDb dosya adı formatı için yaş hesaplama
    parts = filename.split('_')
    birth_year = int(parts[2])
    photo_year = int(parts[3].split('-')[0])
    return photo_year - birth_year

processed_count = 0
skipped_count = 0

for folder_path in folder_paths:
    for image_file in os.listdir(folder_path):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            if image is None:
                skipped_count += 1
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 1)

            if len(faces) == 0:
                skipped_count += 1
                continue

            for face in faces:
                landmarks = predictor(gray, face)
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y

                # Geçersiz yüz bölgesi kontrolü, yoksa hata veriyor
                if w <= 0 or h <= 0:
                    print(f"Geçersiz yüz bölgesi: {image_file}")
                    skipped_count += 1
                    continue

                cropped_face = gray[y:y+h, x:x+w]
                
                # Boş kırpılmış yüz kontrolü
                if cropped_face.size == 0 or cropped_face is None:
                    print(f"Boş kırpılmış yüz: {image_file}")
                    skipped_count += 1
                    continue
                
                
                lbp_hist = extract_lbp_features(cropped_face)


                # Gözler arası ve ağız genişliği mesafesi özellikleri
                eye_distance = np.linalg.norm(np.array([landmarks.part(36).x, landmarks.part(36).y]) - np.array([landmarks.part(45).x, landmarks.part(45).y]))
                mouth_width = np.linalg.norm(np.array([landmarks.part(48).x, landmarks.part(48).y]) - np.array([landmarks.part(54).x, landmarks.part(54).y]))

                feature_vector = np.concatenate(([eye_distance, mouth_width], lbp_hist))
                feature_vectors.append(feature_vector)

                if 'UTKFace' in folder_path:
                    age_labels.append(int(image_file.split('_')[0]))
                elif 'imdb' in folder_path:
                    age_labels.append(calculate_age_from_imdb_filename(image_file))
                
                image_paths.append(image_path)
                processed_count += 1

print(f"Processed: {processed_count}, Skipped: {skipped_count}")

feature_vectors = np.array(feature_vectors)
age_labels = np.array(age_labels)
image_paths = np.array(image_paths)
# Veri setini eğitim ve test setlerine ayırma, %20 % 80
X_train, X_test, y_train, y_test, X_train_paths, X_test_paths = train_test_split(feature_vectors, age_labels, image_paths, test_size=0.2, random_state=42)

# Nearest Neighbors modelini eğitme
flann = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X_train)

predicted_ages = []
prediction_errors = []  # Tahmin hatalarını saklamak için
best_prediction_error = np.inf
worst_prediction_error = -np.inf
best_image_path = None
worst_image_path = None


for i, (test_vector, actual_age) in enumerate(zip(X_test, y_test)):
    test_vector = test_vector.reshape(1, -1)
    _, indices = flann.kneighbors(test_vector)
    predicted_age = y_train[indices[0]].mean()
    predicted_ages.append(predicted_age)
    
    error = abs(predicted_age - actual_age)
    prediction_errors.append(error)
    
    if error < best_prediction_error:
        best_prediction_error = error
        best_image_path = X_test_paths[i]
    if error > worst_prediction_error:
        worst_prediction_error = error
        worst_image_path = X_test_paths[i]

    if i % 100 == 0 or i == len(X_test) - 1:
        print(f"{i+1}/{len(X_test)} samples processed... Actual Age: {actual_age}, Predicted Age: {predicted_age:.2f}")

mae = mean_absolute_error(y_test, predicted_ages)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# En iyi ve en kötü tahminlerin görsellerini gösterme
def show_image(image_path, title):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

if best_image_path and worst_image_path:
    show_image(best_image_path, f'Best Prediction (Actual Age: {y_test[np.argmin(prediction_errors)]}, Predicted Age: {predicted_ages[np.argmin(prediction_errors)]:.2f}, Error: {best_prediction_error})')
    show_image(worst_image_path, f'Worst Prediction (Actual Age: {y_test[np.argmax(prediction_errors)]}, Predicted Age: {predicted_ages[np.argmax(prediction_errors)]:.2f}, Error: {worst_prediction_error})')

# Modeli kaydetme, sonradan kullanılacak
joblib.dump(flann, '/home/sevvalizmirli/Desktop/Computer Vision/Github/Computer-Vision-Human-Age-Detection/flann_model.joblib')

