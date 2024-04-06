import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern

# Kaydedilmiş FLANN modelini yükle
model_path = '/home/sevvalizmirli/Desktop/Computer Vision/Github/Computer-Vision-Human-Age-Detection/model1.joblib'
loaded_model = joblib.load(model_path)

# Gerçek zamanlı veri örneğini yükle ve ön işlemden geçir
def load_and_preprocess_single_image(image_path):
    P = 8  # LBP için numara komşuları
    R = 1  # LBP için yarıçap
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist.reshape(1, -1), image

# Görüntü yolu
real_time_image_path = '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/Test/sevval_25.jpeg'

# Görüntüyü yükle ve ön işlemden geçir
real_time_feature_vector, real_time_image = load_and_preprocess_single_image(real_time_image_path)

# Yüklü modeli kullanarak yaş tahmini yap
neighbors = loaded_model.kneighbors(real_time_feature_vector, n_neighbors=3, return_distance=False)
predicted_ages = [age_labels[n] for n in neighbors[0]]  # age_labels listesi önceki eğitimden alınan yaş etiketleridir
predicted_age = np.mean(predicted_ages)
print(f"Predicted age for real-time image: {predicted_age:.2f}")

# Tahmini yaş değerini görsel üzerine yazdır
cv2.putText(real_time_image, f"Predicted Age: {predicted_age:.1f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Görseli göster
cv2.imshow('Age Prediction', real_time_image)
cv2.waitKey(0)  # Bir tuşa basılmasını bekler
cv2.destroyAllWindows()
