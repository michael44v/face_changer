import numpy as np


LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]


def eye_aspect_ratio(eye):
    eye = np.asarray(
        eye,
        dtype=np.float32
    )

    horizontal = np.linalg.norm(
        eye[0] - eye[3]
    )

    vertical_1 = np.linalg.norm(
        eye[1] - eye[5]
    )

    vertical_2 = np.linalg.norm(
        eye[2] - eye[4]
    )

    if horizontal < 1e-6:
        return 0.0

    return (
        vertical_1 + vertical_2
    ) / (
        2.0 * horizontal
    )


def get_eye_expression(landmarks):

    points = np.asarray(
        landmarks,
        dtype=np.float32
    )

    left_eye = points[LEFT_EYE]
    right_eye = points[RIGHT_EYE]

    left_ear = eye_aspect_ratio(
        left_eye
    )

    right_ear = eye_aspect_ratio(
        right_eye
    )

    average_ear = (
        left_ear + right_ear
    ) / 2.0

    return {
        "left_ear": left_ear,
        "right_ear": right_ear,
        "average_ear": average_ear
    }