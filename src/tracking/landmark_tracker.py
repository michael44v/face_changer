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

        # Smooth the movement.
        tracked = (
            self.smoothing * old_points
            + (1.0 - self.smoothing) * tracked
        )

        tracked[:, 0] = np.clip(
            tracked[:, 0],
            0,
            width - 1
        )

        tracked[:, 1] = np.clip(
            tracked[:, 1],
            0,
            height - 1
        )

        self.previous_gray = gray

        self.previous_landmarks = tracked.copy()

        return tracked

    def correct(self, landmarks):

        if landmarks is None:
            return None

        landmarks = np.asarray(
            landmarks,
            dtype=np.float32
        )

        if self.previous_landmarks is None:

            self.previous_landmarks = landmarks.copy()

            return landmarks

        # Blend newly detected landmarks with
        # the previous stable positions.
        corrected = (
            self.smoothing
            * self.previous_landmarks
            +
            (1.0 - self.smoothing)
            * landmarks
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