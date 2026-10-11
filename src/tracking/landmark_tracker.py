import cv2
import numpy as np


class LandmarkTracker:

    def __init__(self):

        self.previous_gray = None
        self.previous_landmarks = None

        self.smoothing = 0.65

        self.min_points = 45

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS
                | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01
            )
        )

    def initialize(self, frame, landmarks):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        landmarks = np.asarray(
            landmarks,
            dtype=np.float32
        )

        self.previous_gray = gray

        self.previous_landmarks = landmarks.copy()

    def update(self, frame):

        if (
            self.previous_gray is None
            or self.previous_landmarks is None
        ):
            return None

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        old_points = (
            self.previous_landmarks
        )

        old_points_lk = (
            old_points
            .reshape(-1, 1, 2)
        )

        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            old_points_lk,
            None,
            **self.lk_params
        )

        if new_points is None:

            self.reset()

            return None

        new_points = (
            new_points
            .reshape(-1, 2)
        )

        status = status.reshape(-1)

        height, width = gray.shape

        movement = np.linalg.norm(
            new_points - old_points,
            axis=1
        )

        inside = (
            (new_points[:, 0] >= 0)
            & (new_points[:, 0] < width)
            & (new_points[:, 1] >= 0)
            & (new_points[:, 1] < height)
        )

        valid = (
            (status == 1)
            & (movement < 35)
            & inside
        )

        if np.sum(valid) < self.min_points:

            self.reset()

            return None

        tracked = old_points.copy()

        tracked[valid] = new_points[valid]

        # Dynamic adaptive smoothing to eliminate jitter while avoiding lag during fast movement
        dist = np.linalg.norm(tracked - old_points, axis=1, keepdims=True)
        # For small dist (< 3px), high smoothing (0.75 old, 0.25 new)
        # For larger dist (> 10px), lower smoothing (0.25 old, 0.75 new)
        alpha = np.clip(0.75 - (dist / 15.0) * 0.50, 0.20, 0.80)

        smoothed = alpha * old_points + (1.0 - alpha) * tracked

        smoothed[:, 0] = np.clip(
            smoothed[:, 0],
            0,
            width - 1
        )

        smoothed[:, 1] = np.clip(
            smoothed[:, 1],
            0,
            height - 1
        )

        self.previous_gray = gray

        self.previous_landmarks = smoothed.copy()

        return smoothed

    def correct(self, landmarks):

        if landmarks is None:
            return None

        landmarks = np.asarray(
            landmarks,
            dtype=np.float32
        )

        if landmarks.shape != (68, 2) or not np.isfinite(landmarks).all():
            return self.previous_landmarks

        if self.previous_landmarks is None:

            self.previous_landmarks = landmarks.copy()

            return landmarks

        # Validate distance to prevent catastrophic jumps on false detections
        dist = np.linalg.norm(landmarks - self.previous_landmarks, axis=1)
        mean_dist = np.mean(dist)

        # If detected landmarks jump by more than 80px average, reject as false detection
        if mean_dist > 80.0:
            return self.previous_landmarks

        # Dynamic blend based on displacement
        alpha = 0.60 if mean_dist < 15.0 else 0.30
        corrected = (
            alpha * self.previous_landmarks
            + (1.0 - alpha) * landmarks
        )

        self.previous_landmarks = corrected.copy()

        return corrected

    def reset(self):

        self.previous_gray = None
        self.previous_landmarks = None

    def is_tracking(self):

        return (
            self.previous_gray is not None
            and self.previous_landmarks is not None
        )