import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import time

# Start time
start_time = time.time()

dataset = 'UTKFace'  # 'UTKFace', 'imdb', veya 'Both'

folder_paths = []
if dataset in ['UTKFace', 'Both']:
     folder_paths = [
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part1/',
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part2/',
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part3/',
    '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part4/'
    ]
sayac = 0
if dataset in ['imdb', 'Both']:
    base_path = '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/imdb_crop/'
    for subdir in os.listdir(base_path):
        if sayac < 1:
            sayac = sayac + 1
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
    try:
        # Örneğin, dosya adı 'nm3595501_rm3452356234_1984-5-23_2005.jpg' formatında ise:
        parts = filename.split('_')  # '_' ile ayırır
        dob_year = parts[-2].split('-')[0]  # Doğum yılını alır ('1984-5-23' kısmını al ve '-' ile ayırarak yılı seçer)
        photo_year = parts[-1].split('.')[0]  # Fotoğrafın çekildiği yılı alır ('2005.jpg' kısmını alır ve '.' ile ayırarak yılı seçer)
        age = int(photo_year) - int(dob_year)
        return age
    except ValueError as e:
        print(f"Error processing file ({filename}): {e}")
        return None


def simple_face_detector(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image could not be read: {image_path}")
        return None
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 48, 80], dtype="uint8")
    upper_color = np.array([20, 255, 255], dtype="uint8")
    skin_mask = cv2.inRange(hsv_image, lower_color, upper_color)
    if skin_mask.sum() == 0:
        print(f"No face detected: {image_path}")
        return None
    return skin_mask

processed_count = 0
skipped_count = 0

# age_label atamasını günceller
for folder_path in folder_paths:
    for image_file in os.listdir(folder_path):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                skipped_count += 1
                continue
            lbp_hist = extract_lbp_features(image)
            feature_vector = lbp_hist
            feature_vectors.append(feature_vector)

            if 'UTKFace' in folder_path:
                try:
                    age_label = int(image_file.split('_')[0])
                except ValueError:
                    print(f"Invalid age format in UTKFace dataset: {image_file}")
                    skipped_count += 1
                    continue
            else:
                age_label = calculate_age_from_imdb_filename(image_file)
                if age_label is None:
                    skipped_count += 1
                    continue

            age_labels.append(age_label)
            image_paths.append(image_path)
            processed_count += 1
            print(f"Processed: {processed_count}, Skipped: {skipped_count}")

print(f"Total processed: {processed_count}, Total skipped: {skipped_count}")

feature_vectors = np.array(feature_vectors)
age_labels = np.array(age_labels)
image_paths = np.array(image_paths)

X_train, X_test, y_train, y_test, X_train_paths, X_test_paths = train_test_split(
    feature_vectors, age_labels, image_paths, test_size=0.2, random_state=42
)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

# Training
flann = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X_train)

predicted_ages = []
prediction_errors = []  # Tahmin hatalarını saklamak için
best_prediction_error = np.inf
worst_prediction_error = -np.inf
best_image_path = None
worst_image_path = None

print("Tahminlere başlanıyor...")
for i, (test_vector, actual_age) in enumerate(zip(X_test, y_test)):
    test_vector = test_vector.reshape(1, -1)
    distances, indices = flann.kneighbors(test_vector)
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
def show_image(image_path, title):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Matplotlib için RGB formatına çevirir
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

if best_image_path and worst_image_path:
    show_image(best_image_path, f'En İyi Tahmin (Gerçek Yaş: {y_test[np.argmin(prediction_errors)]}, Tahmin Edilen Yaş: {predicted_ages[np.argmin(prediction_errors)]:.2f}, Hata: {best_prediction_error})')
    show_image(worst_image_path, f'En Kötü Tahmin (Gerçek Yaş: {y_test[np.argmax(prediction_errors)]}, Tahmin Edilen Yaş: {predicted_ages[np.argmax(prediction_errors)]:.2f}, Hata: {worst_prediction_error})')

# Modeli kaydeder, sonrası için
joblib.dump(flann, '/home/sevvalizmirli/Desktop/Computer Vision/Github/Computer-Vision-Human-Age-Detection/flann_model.joblib')

# End time
end_time = time.time()


total_time = end_time - start_time
total_time_minutes = total_time / 60  

print(f"Toplam süre: {total_time:.2f} saniye ({total_time_minutes:.2f} dakika)")