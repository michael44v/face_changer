import unittest
import numpy as np
import cv2

from src.expressions.expression_controller import ExpressionController
from src.expressions.expression_deformer import deform_landmarks_for_expression
from src.transformation.target_face import TargetFace
from src.transformation.face_warper import warp_face
from src.blending.blender import blend_face


class TestIntegrationPipeline(unittest.TestCase):

    def setUp(self):
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel("models/lbfmodel.yaml")
        self.target = TargetFace("assets/targets/image.png", self.facemark)

    def test_target_loading(self):
        self.assertIsNotNone(self.target.get_image())
        landmarks = self.target.get_landmarks()
        self.assertIsNotNone(landmarks)
        self.assertEqual(landmarks.shape, (68, 2))

    def test_end_to_end_blink_rendering(self):
        target_img = self.target.get_image()
        target_lm = self.target.get_landmarks()

        controller = ExpressionController()

        # Simulate open eyes
        open_eye_data = {
            "left_ear": 0.32,
            "right_ear": 0.32,
            "mouth_open": 0.05,
            "smile": 0.1,
            "eyebrow_elevation": 0.0,
        }
        expr_open = controller.update(open_eye_data)

        # Simulate closed eyes
        closed_eye_data = {
            "left_ear": 0.10,
            "right_ear": 0.10,
            "mouth_open": 0.05,
            "smile": 0.1,
            "eyebrow_elevation": 0.0,
        }
        for _ in range(3):
            expr_closed = controller.update(closed_eye_data)

        self.assertTrue(expr_closed["left_closed"])
        self.assertTrue(expr_closed["right_closed"])

        # Deform landmarks
        deformed_target_lm = deform_landmarks_for_expression(
            target_lm, expr_closed, is_target_source=True
        )
        deformed_dest_lm = deform_landmarks_for_expression(
            target_lm, expr_closed, is_target_source=False
        )

        self.assertEqual(deformed_target_lm.shape, (68, 2))
        self.assertEqual(deformed_dest_lm.shape, (68, 2))

        # Perform warp & blend
        warped = warp_face(
            target_img, deformed_target_lm, target_img.shape, deformed_dest_lm
        )
        blended = blend_face(target_img, warped, deformed_dest_lm)

        self.assertEqual(blended.shape, target_img.shape)
        self.assertEqual(blended.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
