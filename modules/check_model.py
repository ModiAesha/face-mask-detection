import cv2
import numpy as np
from tensorflow.keras.models import load_model
from face_detect import FaceDetector

IMG_SIZE = 128
CLASS_NAMES = ["mask_improper", "mask_proper", "no_mask"]  # matches Colab's alphabetical class_indices order

model = load_model("../model/mask_detector_model.h5")
print("Model loaded successfully")

detector = FaceDetector()
img = cv2.imread("../images/WIN_20260925_08_53_48_Pro.jpg")
faces = detector.detect_faces(img)
print(f"Detected {len(faces)} face(s)")

if len(faces) == 0:
    print("No face found, cannot classify.")
else:
    (x, y, w, h) = faces[0]
    face_crop = img[y:y+h, x:x+w]

    face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)  # OpenCV loads as BGR, model trained on RGB
    face_normalized = face_rgb / 255.0
    face_batch = np.expand_dims(face_normalized, axis=0)

    predictions = model.predict(face_batch)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[predicted_index] * 100

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")