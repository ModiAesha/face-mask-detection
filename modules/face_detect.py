#============= DNN detector ===============
import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
        self.confidence_threshold = 0.5 

    def detect_faces(self, image):
        h, w = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image,
            1.0,
            (400, 400),
            (104.0, 177.0, 123.0)
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])

            if confidence > 0.5:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                box_w = endX - startX
                box_h = endY - startY

                if box_w > 0 and box_h > 0:

                    print(
                        f"Detection {i}: "
                        f"confidence={confidence:.3f}, "
                        f"box=({startX}, {startY}, {endX}, {endY})"
                    )

                    cv2.rectangle(
                        image,
                        (startX, startY),
                        (endX, endY),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        image,
                        f"{confidence:.2f}",
                        (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                    faces.append((startX, startY, box_w, box_h))

        return faces

    def crop_faces(self, image, faces):
        """
        Returns list of cropped face images (BGR) matching each bounding box
        """
        crops = []
        for (x, y, w, h) in faces:
            crops.append(image[y:y+h, x:x+w])
        return crops

