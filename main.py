import time
import cv2
import numpy as np

from src.transformation.target_face import TargetFace
from src.transformation.face_warper import warp_face
from src.blending.blender import blend_face
from src.tracking.landmark_tracker import LandmarkTracker
from src.expressions.eye_expression import get_facial_expressions
from src.expressions.expression_controller import ExpressionController
from src.expressions.expression_deformer import deform_landmarks_for_expression


def main():
    # --------------------------------------------------
    # Face detector
    # --------------------------------------------------
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # --------------------------------------------------
    # 68-point landmark model
    # --------------------------------------------------
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel("models/lbfmodel.yaml")

    # --------------------------------------------------
    # Target face
    # --------------------------------------------------
    target = TargetFace("assets/targets/image.png", facemark)
    target_image = target.get_image()
    target_landmarks = target.get_landmarks()

    if target_image is None or target_landmarks is None:
        print("[ERROR] Failed to load target image or extract target landmarks.")
        return

    # --------------------------------------------------
    # Camera
    # --------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WARNING] Could not open webcam at index 0. Check camera connection.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --------------------------------------------------
    # Tracking & State
    # --------------------------------------------------
    tracker = LandmarkTracker()
    expression_controller = ExpressionController()

    frames_since_detection = 0
    REDETECT_INTERVAL = 5
    show_debug = True

    prev_time = time.time()
    fps = 0.0

    print("\n--- Real-Time Face Transformer ---")
    print("  'q' - Quit")
    print("  'd' - Toggle Debug Overlay")
    print("  'r' - Reset Tracker & Expression Calibration\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] No camera frame received. Exiting main loop.")
            break

        frame = cv2.flip(frame, 1)
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
        prev_time = curr_time

        landmarks = None

        # ----------------------------------------------
        # 1. Optical Flow Tracking
        # ----------------------------------------------
        if tracker.is_tracking():
            landmarks = tracker.update(frame)

        # ----------------------------------------------
        # 2. Periodic Face Redetection
        # ----------------------------------------------
        frames_since_detection += 1
        should_redetect = (
            landmarks is None or frames_since_detection >= REDETECT_INTERVAL
        )

        if should_redetect:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            if len(faces) > 0:
                success, detected_landmarks = facemark.fit(frame, faces)
                if success and len(detected_landmarks) > 0:
                    raw_lm = detected_landmarks[0][0]
                    landmarks = tracker.correct(raw_lm)
                    tracker.initialize(frame, landmarks)
                    frames_since_detection = 0
            else:
                if landmarks is None:
                    tracker.reset()
                    expression_controller.reset()

        # ----------------------------------------------
        # 3. Expression Transfer & Transformation
        # ----------------------------------------------
        display_frame = frame.copy()

        if landmarks is not None:
            # Measure facial expression parameters (eyes, mouth, eyebrows)
            facial_data = get_facial_expressions(landmarks)
            expression = expression_controller.update(facial_data)

            # Deform source target image landmarks (sampling skin above eye on blinks)
            deformed_target_landmarks = deform_landmarks_for_expression(
                target_landmarks, expression, is_target_source=True
            )

            # Deform destination output landmarks (collapsing eye polygons on blinks)
            deformed_dest_landmarks = deform_landmarks_for_expression(
                landmarks, expression, is_target_source=False
            )

            # Warp target face onto destination coordinates
            warped_face = warp_face(
                target_image,
                deformed_target_landmarks,
                frame.shape,
                deformed_dest_landmarks,
            )

            # Feathered convex hull blend
            transformed = blend_face(frame, warped_face, deformed_dest_landmarks)
            display_frame = transformed

            # Render Debug Overlay if toggled on
            if show_debug:
                # Eye contour lines
                eye_contours = [list(range(36, 42)), list(range(42, 48))]
                for contour in eye_contours:
                    pts = landmarks[contour].astype("int32").reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [pts], True, (0, 255, 0), 1)

                # Eye landmark points
                for i in range(36, 48):
                    pt = tuple(landmarks[i].astype(int))
                    cv2.circle(display_frame, pt, 2, (0, 0, 255), -1)

                # Diagnostic telemetry text
                lines = [
                    f"FPS: {fps:.1f}",
                    f"EAR Raw L:{facial_data['left_ear']:.3f} R:{facial_data['right_ear']:.3f}",
                    f"Closure L:{expression['left_closure']:.2f} R:{expression['right_closure']:.2f}",
                    f"Base L:{expression['left_baseline']:.3f} R:{expression['right_baseline']:.3f}",
                    f"Mouth Open:{expression['mouth_open']:.2f} Eyebrow:{expression['eyebrow_elevation']:.2f}",
                    f"Calibrated: {expression['calibrated']}",
                    f"Closed L:{expression['left_closed']} R:{expression['right_closed']}",
                    f"Blinks: {expression['blink_count']}",
                ]

                for idx, text in enumerate(lines):
                    cv2.putText(
                        display_frame,
                        text,
                        (10, 25 + idx * 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if expression["blink_detected"]:
                    cv2.putText(
                        display_frame,
                        "BLINK DETECTED!",
                        (10, 195),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

        cv2.imshow("Real-Time Face Transformer", display_frame)

        # ----------------------------------------------
        # Key Controls
        # ----------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            show_debug = not show_debug
            print(f"[INFO] Debug overlay: {'ON' if show_debug else 'OFF'}")
        elif key == ord("r"):
            tracker.reset()
            expression_controller.reset()
            print("[INFO] Reset tracker and expression calibration.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
