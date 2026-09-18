import cv2
import numpy as np


def rect_contains(rect, point):
    x, y, w, h = rect

    return (
        x <= point[0] < x + w
        and y <= point[1] < y + h
    )


def calculate_delaunay_triangles(rect, points):
    x, y, w, h = rect

    points = np.asarray(
        points,
        dtype=np.float32
    ).copy()

    # Keep every landmark inside the destination frame.
    points[:, 0] = np.clip(
        points[:, 0],
        x,
        x + w - 1
    )

    points[:, 1] = np.clip(
        points[:, 1],
        y,
        y + h - 1
    )

    subdiv = cv2.Subdiv2D(rect)

    for point in points:

        try:
            subdiv.insert(
                (
                    float(point[0]),
                    float(point[1])
                )
            )

        except cv2.error:
            continue

    triangle_list = subdiv.getTriangleList()

    triangles = []

    for triangle in triangle_list:

        triangle_points = [
            (triangle[0], triangle[1]),
            (triangle[2], triangle[3]),
            (triangle[4], triangle[5])
        ]

        if not all(
            rect_contains(rect, point)
            for point in triangle_points
        ):
            continue

        indices = []

        for point in triangle_points:

            distances = np.linalg.norm(
                points - np.array(point),
                axis=1
            )

            index = np.argmin(distances)

            if distances[index] < 2.0:
                indices.append(index)

        if (
            len(indices) == 3
            and len(set(indices)) == 3
        ):

            triangle_indices = tuple(indices)

            if triangle_indices not in triangles:
                triangles.append(
                    triangle_indices
                )

    return triangles

def warp_triangle(
    source,
    destination,
    source_triangle,
    destination_triangle
):

    source_triangle = np.float32(
        source_triangle
    )

    destination_triangle = np.float32(
        destination_triangle
    )

    source_rect = cv2.boundingRect(
        source_triangle
    )

    destination_rect = cv2.boundingRect(
        destination_triangle
    )

    sx, sy, sw, sh = source_rect
    dx, dy, dw, dh = destination_rect

    if (
        sw <= 0
        or sh <= 0
        or dw <= 0
        or dh <= 0
    ):
        return

    source_crop = source[
        sy:sy + sh,
        sx:sx + sw
    ]

    if source_crop.size == 0:
        return

    source_local = (
        source_triangle
        - np.array(
            [sx, sy],
            dtype=np.float32
        )
    )

    destination_local = (
        destination_triangle
        - np.array(
            [dx, dy],
            dtype=np.float32
        )
    )

    matrix = cv2.getAffineTransform(
        source_local,
        destination_local
    )

    warped = cv2.warpAffine(
        source_crop,
        matrix,
        (dw, dh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    mask = np.zeros(
        (dh, dw),
        dtype=np.uint8
    )

    cv2.fillConvexPoly(
        mask,
        np.int32(destination_local),
        255
    )

    destination_region = destination[
        dy:dy + dh,
        dx:dx + dw
    ]

    if destination_region.shape[:2] != mask.shape:
        return

    mask_bool = mask > 0

    destination_region[mask_bool] = (
        warped[mask_bool]
    )


def warp_face(
    target_image,
    target_landmarks,
    destination_shape,
    destination_landmarks
):

    height, width = destination_shape[:2]

    target_points = np.asarray(
        target_landmarks,
        dtype=np.float32
    )

    destination_points = np.asarray(
        destination_landmarks,
        dtype=np.float32
    )

    rect = (
        0,
        0,
        width,
        height
    )

    triangles = calculate_delaunay_triangles(
        rect,
        destination_points
    )

    # Separate canvas for the transformed face
    warped_face = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    for triangle in triangles:

        source_triangle = [
            target_points[i]
            for i in triangle
        ]

        destination_triangle = [
            destination_points[i]
            for i in triangle
        ]

        warp_triangle(
            target_image,
            warped_face,
            source_triangle,
            destination_triangle
        )

    return warped_face