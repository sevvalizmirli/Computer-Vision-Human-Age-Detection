import cv2
import numpy as np
import joblib
from skimage.feature import hog
import matplotlib.pyplot as plt


pca = joblib.load('pca.joblib')
knn_model = joblib.load('knn_model.joblib')

def preprocess_and_predict(image_path, pca, knn_model, img_size=(128, 128)):

    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, img_size)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # HOG özniteliklerini çıkarır
    hog_feature, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, feature_vector=True)
    
    # HOG özniteliklerini PCA ile boyutunu indirger
    hog_feature_pca = pca.transform([hog_feature])
    
   
    predicted_age = knn_model.predict(hog_feature_pca)[0]
    
  
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted Age: {int(predicted_age)}")
    plt.show()


image_path = '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/Test/Oguz_Ergin.jpeg'


preprocess_and_predict(image_path, pca, knn_model)
