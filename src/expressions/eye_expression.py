import numpy as np


def _eye_aspect_ratio(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR) from six 2D eye landmarks.

    Points ordering for standard 68-landmark model:
      0: outer corner (p1)
      1: upper eyelid left (p2)
      2: upper eyelid right (p3)
      3: inner corner (p4)
      4: lower eyelid right (p5)
      5: lower eyelid left (p6)

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    eye = np.asarray(eye_points, dtype=np.float64)

    if eye.shape != (6, 2):
        return 0.0

    if not np.all(np.isfinite(eye)):
        return 0.0

    vertical_1 = float(np.linalg.norm(eye[1] - eye[5]))
    vertical_2 = float(np.linalg.norm(eye[2] - eye[4]))
    horizontal = float(np.linalg.norm(eye[0] - eye[3]))

    if horizontal < 1e-6:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(np.clip(ear, 0.0, 1.0))


def get_eye_expression(landmarks):
    """
    Extract Eye Aspect Ratio (EAR) for left and right eyes from 68-point landmarks.

    Returns:
        {
            "left_ear": float,
            "right_ear": float,
            "valid": bool
        }
    """
    if landmarks is None:
        return {"left_ear": 0.0, "right_ear": 0.0, "valid": False}

    points = np.asarray(landmarks, dtype=np.float64)

    if points.ndim != 2 or points.shape[0] != 68 or points.shape[1] < 2:
        return {"left_ear": 0.0, "right_ear": 0.0, "valid": False}

    if not np.all(np.isfinite(points)):
        return {"left_ear": 0.0, "right_ear": 0.0, "valid": False}

    points = points[:, :2]

    left_eye = points[36:42]
    right_eye = points[42:48]

    left_ear = _eye_aspect_ratio(left_eye)
    right_ear = _eye_aspect_ratio(right_eye)

    valid = (left_ear > 0.05 and right_ear > 0.05)

    return {
        "left_ear": left_ear,
        "right_ear": right_ear,
        "valid": valid
    }


def get_facial_expressions(landmarks):
    """
    Extract facial expression measurements from 68-point landmarks:
      - eye EARs
      - mouth opening ratio
      - smile ratio
      - eyebrow elevation ratio
    """
    eye_data = get_eye_expression(landmarks)
    if not eye_data["valid"]:
        return {
            **eye_data,
            "mouth_open": 0.0,
            "smile": 0.0,
            "eyebrow_elevation": 0.0,
        }

    points = np.asarray(landmarks, dtype=np.float64)[:, :2]

    # Mouth opening calculation using inner lip landmarks (60-67)
    # 60: left corner, 64: right corner, 62: top inner, 66: bottom inner
    mouth_width = float(np.linalg.norm(points[48] - points[54]))
    inner_height_1 = float(np.linalg.norm(points[61] - points[67]))
    inner_height_2 = float(np.linalg.norm(points[62] - points[66]))
    inner_height_3 = float(np.linalg.norm(points[63] - points[65]))
    mouth_height = (inner_height_1 + inner_height_2 + inner_height_3) / 3.0

    mouth_open = mouth_height / max(mouth_width, 1e-6)

    # Smile calculation: ratio of mouth width to distance between cheek outer edges (landmark 2 to 14)
    face_width = float(np.linalg.norm(points[2] - points[14]))
    smile = mouth_width / max(face_width, 1e-6)

    # Eyebrow elevation: average distance from eyebrow center to eye center relative to eye distance
    left_brow_center = np.mean(points[17:22], axis=0)
    right_brow_center = np.mean(points[22:27], axis=0)
    left_eye_center = np.mean(points[36:42], axis=0)
    right_eye_center = np.mean(points[42:48], axis=0)

    left_brow_dist = float(np.linalg.norm(left_brow_center - left_eye_center))
    right_brow_dist = float(np.linalg.norm(right_brow_center - right_eye_center))
    eye_dist = float(np.linalg.norm(right_eye_center - left_eye_center))

    eyebrow_elevation = ((left_brow_dist + right_brow_dist) / 2.0) / max(eye_dist, 1e-6)

    return {
        "left_ear": eye_data["left_ear"],
        "right_ear": eye_data["right_ear"],
        "valid": True,
        "mouth_open": float(np.clip(mouth_open, 0.0, 1.5)),
        "smile": float(np.clip(smile, 0.0, 1.0)),
        "eyebrow_elevation": float(np.clip(eyebrow_elevation, 0.0, 1.0)),
    }
