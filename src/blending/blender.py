import cv2
import numpy as np


def blend_face(
    frame,
    warped_face,
    landmarks
):

    height, width = frame.shape[:2]

    points = np.asarray(
        landmarks,
        dtype=np.float32
    )

    points = np.round(points).astype(
        np.int32
    )

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

    # Create face mask
    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    hull = cv2.convexHull(points)

    cv2.fillConvexPoly(
        mask,
        hull,
        255
    )

    # Soften the edge
    mask = cv2.GaussianBlur(
        mask,
        (31, 31),
        0
    )

    # Normalize mask
    mask_float = (
        mask.astype(np.float32)
        / 255.0
    )

    mask_float = mask_float[:, :, None]

    # Blend transformed face with webcam
    result = (
        warped_face.astype(np.float32)
        * mask_float
        +
        frame.astype(np.float32)
        * (1.0 - mask_float)
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)