import cv2
import dlib

# Dosya yolunu bir değişkende sakla
image_path = '/home/sevvalizmirli/Desktop/Computer Vision/Dataset/UTKFace_part1/1_0_0_20161219140623097.jpg'

# Görüntüyü yükle
image = cv2.imread(image_path)

if image is None:
    print(f"Dosya yolu doğru ancak dosya okunamıyor veya bulunamıyor: {image_path}")
else:
    # Gürültüyü azalt
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Yüz tespit ediciyi yükle
    detector = dlib.get_frontal_face_detector()

    # Yüzleri tespit et
    detected_faces = detector(blurred_image, 1)

    # Tespit edilen yüzlerin etrafına dikdörtgen çiz
    for face in detected_faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Görüntüyü göster
    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
