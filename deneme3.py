import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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

for folder_path in folder_paths:
    for image_file in os.listdir(folder_path):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Histogram Equalization
            gray = cv2.equalizeHist(gray)
            
           
            faces, _ = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

            if len(faces) == 0:
                skipped_count += 1
                continue

            for (x, y, w, h) in faces:
                cropped_face = gray[y:y+h, x:x+w]
                if cropped_face.size == 0:
                    skipped_count += 1
                    continue

                lbp_hist = extract_lbp_features(cropped_face)
                feature_vector = lbp_hist
                feature_vectors.append(feature_vector)
                age_labels.append(int(image_file.split('_')[0]))
                processed_count += 1

print(f"Processed file count: {processed_count}")
print(f"Skipped file count: {skipped_count}")

feature_vectors = np.array(feature_vectorsage_labels = np.array(age_labels))

X_train, X_test, y_train, y_test = train_test_split(feature_vectors, age_labels, test_size=0.2, random_state=42)

flann = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X_train)

predicted_ages = []
for i, (test_vector, actual_age) in enumerate(zip(X_test, y_test)):
    test_vector = test_vector.reshape(1, -1)
    _, indices = flann.kneighbors(test_vector)
    predicted_age = y_train[indices[0]].mean()
    predicted_ages.append(predicted_age)
    if i % 100 == 0 or i == len(X_test) - 1:
        print(f"{i+1}/{len(X_test)} samples processed... Actual Age: {actual_age}, Predicted Age: {predicted_age:.2f}")

mae = mean_absolute_error(y_test, predicted_ages)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Displaying detailed results for the first 20 test samples
detailed_results = zip(y_test, predicted_ages)
for index, (actual, predicted) in enumerate(detailed_results):
    if index < 20:
        print(f"Test Sample {index + 1} - Actual Age: {actual}, Predicted Age: {predicted:.2f}")

