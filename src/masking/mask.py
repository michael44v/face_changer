import cv2
import numpy as np


def create_face_mask(landmarks, image_size):

    height, width = image_size[:2]

    points = np.asarray(
        landmarks,
        dtype=np.float32
    )

    points = np.round(points).astype(np.int32)

    # Keep points inside image
    points[:, 0] = np.clip(
        points[:, 0],
        0,
        width - 1
    )

    points[:, 1] = np.clip(
        points[:, 1],
        0,
        height - 1
    )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # Convex hull around all facial landmarks
    hull = cv2.convexHull(points)

    cv2.fillConvexPoly(
        mask,
        hull,
        255
    )

    # Smooth the edges
    mask = cv2.GaussianBlur(
        mask,
        (21, 21),
        0
    )

    return mask