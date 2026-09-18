import cv2
import numpy as np


class TargetFace:
    def __init__(self, image_path, facemark):
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise FileNotFoundError(
                f"Could not load target image: {image_path}"
            )

        self.facemark = facemark

        self.landmarks = self._detect_landmarks()

        if self.landmarks is None:
            raise RuntimeError(
                "Could not detect a face in the target image."
            )

    def _detect_landmarks(self):
        gray = cv2.cvtColor(
            self.image,
            cv2.COLOR_BGR2GRAY
        )

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            return None

        success, landmarks = self.facemark.fit(
            self.image,
            faces
        )

        if not success:
            return None

        return landmarks[0][0]

    def get_image(self):
        return self.image

    def get_landmarks(self):
        return self.landmarks