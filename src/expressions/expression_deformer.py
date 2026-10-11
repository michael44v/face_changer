import numpy as np


def deform_landmarks_for_expression(landmarks, expression_data, is_target_source=False):
    """
    Deform face 68 landmarks based on continuous expression parameters from source face:
      - left_closure: 0.0 (open) to 1.0 (closed)
      - right_closure: 0.0 (open) to 1.0 (closed)
      - mouth_open: relative mouth opening ratio (0.0 to ~0.8)
      - eyebrow_elevation: relative eyebrow elevation delta (-0.2 to +0.3)

    If is_target_source=True, adjusts source image landmark sampling coordinates
    so upper eyelid skin is drawn down over closed eye.
    """
    if landmarks is None:
        return None

    pts = np.array(landmarks, dtype=np.float32).copy()
    if pts.shape[0] != 68 or pts.shape[1] < 2:
        return landmarks

    left_closure = float(np.clip(expression_data.get("left_closure", 0.0), 0.0, 1.0))
    right_closure = float(np.clip(expression_data.get("right_closure", 0.0), 0.0, 1.0))

    if is_target_source:
        # For source target image: when eye closes, pull upper eyelid points (37,38,43,44) UP
        # so upper eyelid skin texture is sampled and stretched down to cover closed eyes
        if left_closure > 0.01:
            eye_h = float(np.linalg.norm(pts[37] - pts[41]) + np.linalg.norm(pts[38] - pts[40])) / 2.0
            pts[37, 1] -= left_closure * 0.70 * eye_h
            pts[38, 1] -= left_closure * 0.70 * eye_h

        if right_closure > 0.01:
            eye_h = float(np.linalg.norm(pts[43] - pts[47]) + np.linalg.norm(pts[44] - pts[46])) / 2.0
            pts[43, 1] -= right_closure * 0.70 * eye_h
            pts[44, 1] -= right_closure * 0.70 * eye_h

        return pts

    # ==========================================
    # Destination landmark deformation (Screen coordinates)
    # ==========================================

    # 1. Left Eye Closure (36-41)
    if left_closure > 0.01:
        # Upper eyelid points (37, 38) move DOWN to meet lower eyelid points (41, 40)
        target_37 = 0.90 * pts[41] + 0.10 * pts[37]
        target_38 = 0.90 * pts[40] + 0.10 * pts[38]

        pts[37] = (1.0 - left_closure) * pts[37] + left_closure * target_37
        pts[38] = (1.0 - left_closure) * pts[38] + left_closure * target_38

        # Lower eyelid (40, 41) moves slightly UP
        target_41 = 0.80 * pts[41] + 0.20 * pts[37]
        target_40 = 0.80 * pts[40] + 0.20 * pts[38]

        pts[41] = (1.0 - left_closure) * pts[41] + left_closure * target_41
        pts[40] = (1.0 - left_closure) * pts[40] + left_closure * target_40

    # 2. Right Eye Closure (42-47)
    if right_closure > 0.01:
        target_43 = 0.90 * pts[47] + 0.10 * pts[43]
        target_44 = 0.90 * pts[46] + 0.10 * pts[44]

        pts[43] = (1.0 - right_closure) * pts[43] + right_closure * target_43
        pts[44] = (1.0 - right_closure) * pts[44] + right_closure * target_44

        target_47 = 0.80 * pts[47] + 0.20 * pts[43]
        target_46 = 0.80 * pts[46] + 0.20 * pts[44]

        pts[47] = (1.0 - right_closure) * pts[47] + right_closure * target_47
        pts[46] = (1.0 - right_closure) * pts[46] + right_closure * target_46

    # 3. Mouth Opening (48-67 & Jaw 6-10)
    mouth_open = float(expression_data.get("mouth_open", 0.0))
    if mouth_open > 0.15:
        mouth_width = float(np.linalg.norm(pts[48] - pts[54]))
        shift = (mouth_open - 0.15) * 0.40 * mouth_width

        lower_lip_indices = [55, 56, 57, 58, 59, 65, 66, 67]
        for idx in lower_lip_indices:
            pts[idx, 1] += shift

        jaw_indices = [6, 7, 8, 9, 10]
        for idx in jaw_indices:
            pts[idx, 1] += shift * 0.5

    # 4. Eyebrow Elevation (17-26)
    eyebrow_elevation = float(expression_data.get("eyebrow_elevation", 0.0))
    if abs(eyebrow_elevation) > 0.05:
        eye_dist = float(np.linalg.norm(pts[36] - pts[45]))
        shift = eyebrow_elevation * 0.25 * eye_dist

        brow_indices = list(range(17, 27))
        for idx in brow_indices:
            pts[idx, 1] -= shift

    return pts
