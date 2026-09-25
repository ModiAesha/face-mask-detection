import cv2
import numpy as np
from tensorflow.keras.models import load_model
from face_detect import FaceDetector
from compliance_score import score_face, score_frame

IMG_SIZE = 128
CLASS_NAMES = ["mask_improper", "mask_proper", "no_mask"]  
MODEL_PATH = "../model/mask_detector_model.h5"

STATUS_COLORS = {
    "SAFE": (0, 200, 0),        
    "VIOLATION": (0, 0, 255),  
    "UNCERTAIN": (0, 200, 200) 
}

print("Loading model...")
model = load_model(MODEL_PATH)
detector = FaceDetector()
print("Model and detector ready.")


def classify_face(face_crop):
    face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb / 255.0
    face_batch = np.expand_dims(face_normalized, axis=0)

    predictions = model.predict(face_batch, verbose=0)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100   # <-- cast to plain float here

    return predicted_class, confidence

def process_frame(img):
    """
    Takes an OpenCV BGR image (numpy array), returns:
    - annotated image (numpy array, BGR)
    - list of per-face results
    - frame-level summary dict
    """
    faces = detector.detect_faces(img)
    face_results = []

    for (x, y, w, h) in faces:
        face_crop = img[y:y+h, x:x+w]
        predicted_class, confidence = classify_face(face_crop)
        result = score_face(predicted_class, confidence)
        face_results.append(result)

        color = STATUS_COLORS[result["status"]]
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

        label_line1 = f"{result['predicted_class']} ({result['confidence']}%)"
        label_line2 = f"Score: {result['compliance_score']} | {result['status']}"
        cv2.putText(img, label_line1, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(img, label_line2, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    frame_summary = score_frame(face_results) if face_results else None
    return img, face_results, frame_summary


def process_image(image_path, output_path):
    """CLI-friendly wrapper: reads from disk, processes, saves, prints."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not read image at {image_path}")
        return

    annotated_img, face_results, frame_summary = process_frame(img)

    print(f"Detected {len(face_results)} face(s)")
    for result in face_results:
        print(f"Prediction: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Compliance Score: {result['compliance_score']}/100")
        print(f"Status: {result['status']}")
        print("---")

    if frame_summary:
        print("Frame Summary:", frame_summary)

    success = cv2.imwrite(output_path, annotated_img)
    if success:
        print(f"Saved annotated result to {output_path}")
    else:
        print("ERROR: Failed to save output image.")

