import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern
from sklearn.neighbors import NearestNeighbors

def extract_lbp_features(image, P=8, R=1):
    lbp = local_binary_pattern(image, P=P, R=R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# Modeli ve eğitim etiketlerini yükler
flann = joblib.load('/home/sevvalizmirli/Desktop/Computer Vision/Github/Computer-Vision-Human-Age-Detection/flann_model.joblib')
y_train = np.load('/home/sevvalizmirli/Desktop/Computer Vision/Github/Computer-Vision-Human-Age-Detection/y_train.npy')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(32, 32), scale=1.05)

    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        lbp_features = extract_lbp_features(face_region).reshape(1, -1)
        
        # Yaş tahmini yapar
        _, indices = flann.kneighbors(lbp_features)
        predicted_age = np.mean(y_train[indices[0]])

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Age: {int(predicted_age)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Real-Time Age Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
