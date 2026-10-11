
import numpy as np


def _eye_aspect_ratio(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR) from six eye landmarks.

    Lower EAR generally means the eye is more closed.
    """
    eye = np.asarray(eye_points, dtype=np.float64)

    if eye.shape != (6, 2):
        raise ValueError(
            "Expected six 2D landmarks for each eye."
        )

    vertical_1 = np.linalg.norm(eye[1] - eye[5])
    vertical_2 = np.linalg.norm(eye[2] - eye[4])
    horizontal = np.linalg.norm(eye[0] - eye[3])

    if horizontal < 1e-6:
        return 0.0

    return float(
        (vertical_1 + vertical_2) / (2.0 * horizontal)
    )


def get_eye_expression(landmarks):
    """
    Compatible with the 68-point facial landmark model.

    Returns:
        {
            "left_ear": float,
            "right_ear": float
        }
    """
    points = np.asarray(landmarks, dtype=np.float64)

    if points.shape[0] != 68 or points.shape[1] < 2:
        raise ValueError(
            "Expected 68 facial landmarks with x and y coordinates."
        )

    points = points[:, :2]

    left_eye = points[36:42]
    right_eye = points[42:48]

    return {
        "left_ear": _eye_aspect_ratio(left_eye),
        "right_ear": _eye_aspect_ratio(right_eye),
    }
