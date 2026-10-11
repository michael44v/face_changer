import cv2
import numpy as np

from src.transformation.target_face import TargetFace
from src.transformation.face_warper import warp_face
from src.expressions.expression_deformer import deform_landmarks_for_expression


def main():
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel("models/lbfmodel.yaml")

    target = TargetFace("assets/targets/image.png", facemark)
    target_img = target.get_image()
    target_landmarks = target.get_landmarks()

    dest_shape = target_img.shape
    dest_landmarks = target_landmarks.copy()

    # Open eye
    warped_open = warp_face(
        target_img,
        target_landmarks,
        dest_shape,
        dest_landmarks
    )

    expression_data = {
        "left_closure": 1.0,
        "right_closure": 1.0,
        "mouth_open": 0.0,
        "eyebrow_elevation": 0.0
    }

    # Deform both target source landmarks and destination landmarks
    src_deformed = deform_landmarks_for_expression(target_landmarks, expression_data, is_target_source=True)
    dest_deformed = deform_landmarks_for_expression(dest_landmarks, expression_data, is_target_source=False)

    warped_closed = warp_face(
        target_img,
        src_deformed,
        dest_shape,
        dest_deformed
    )

    h, w = target_img.shape[:2]
    crop_open = warped_open[int(h*0.3):int(h*0.55), int(w*0.25):int(w*0.75)]
    crop_closed = warped_closed[int(h*0.3):int(h*0.55), int(w*0.25):int(w*0.75)]

    combined = np.hstack([crop_open, crop_closed])
    small = cv2.resize(combined, (960, 320))
    cv2.imwrite("test_blink_crop2.png", small)
    print("Saved test_blink_crop2.png successfully!")


if __name__ == "__main__":
    main()
