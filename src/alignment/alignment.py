import cv2
import numpy as np


class FaceAligner:
    def __init__(self, output_size=256):
        self.output_size = output_size

        # Previous values used for smoothing
        self.previous_angle = None
        self.previous_scale = None
        self.previous_center = None

        # Smoothing strength
        self.smoothing = 0.85

    def _smooth(self, current, previous):
        if previous is None:
            return current

        return (
            self.smoothing * previous
            + (1.0 - self.smoothing) * current
        )

    def align(self, frame, landmarks):

        points = np.asarray(
            landmarks,
            dtype=np.float32
        )

        # -----------------------------
        # Eye centers
        # -----------------------------

        left_eye = points[36:42].mean(axis=0)
        right_eye = points[42:48].mean(axis=0)

        eye_center = (
            left_eye + right_eye
        ) / 2.0

        # -----------------------------
        # Calculate rotation
        # -----------------------------

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        angle = np.degrees(
            np.arctan2(dy, dx)
        )

        # -----------------------------
        # Calculate scale
        # -----------------------------

        eye_distance = np.linalg.norm(
            right_eye - left_eye
        )

        if eye_distance < 1:
            return None, None

        desired_eye_distance = 80.0

        scale = (
            desired_eye_distance
            / eye_distance
        )

        # -----------------------------
        # Smooth values
        # -----------------------------

        angle = self._smooth(
            angle,
            self.previous_angle
        )

        scale = self._smooth(
            scale,
            self.previous_scale
        )

        eye_center = self._smooth(
            eye_center,
            self.previous_center
        )

        self.previous_angle = angle
        self.previous_scale = scale
        self.previous_center = eye_center

        # -----------------------------
        # Transformation
        # -----------------------------

        transform_matrix = cv2.getRotationMatrix2D(
            tuple(eye_center),
            angle,
            scale
        )

        desired_eye_center = np.array(
            [
                self.output_size * 0.5,
                self.output_size * 0.38
            ],
            dtype=np.float32
        )

        transform_matrix[0, 2] += (
            desired_eye_center[0]
            - eye_center[0]
        )

        transform_matrix[1, 2] += (
            desired_eye_center[1]
            - eye_center[1]
        )

        # -----------------------------
        # Warp face
        # -----------------------------

        aligned_face = cv2.warpAffine(
            frame,
            transform_matrix,
            (
                self.output_size,
                self.output_size
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return aligned_face, transform_matrix