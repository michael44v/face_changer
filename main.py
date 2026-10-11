import cv2

from src.transformation.target_face import TargetFace
from src.transformation.face_warper import warp_face
from src.blending.blender import blend_face
from src.tracking.landmark_tracker import LandmarkTracker
from src.expressions.eye_expression import get_eye_expression
from src.expressions.expression_controller import ExpressionController


# --------------------------------------------------
# Face detector
# --------------------------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


# --------------------------------------------------
# 68-point landmark model
# --------------------------------------------------

facemark = cv2.face.createFacemarkLBF()

facemark.loadModel(
    "models/lbfmodel.yaml"
)


# --------------------------------------------------
# Target face
# --------------------------------------------------

target = TargetFace(
    "assets/targets/image.png",
    facemark
)

target_image = target.get_image()
target_landmarks = target.get_landmarks()


# --------------------------------------------------
# Camera
# --------------------------------------------------

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# --------------------------------------------------
# Tracking
# --------------------------------------------------

tracker = LandmarkTracker()

expression_controller = ExpressionController()


# --------------------------------------------------
# Detection interval
# --------------------------------------------------

frames_since_detection = 0

REDETECT_INTERVAL = 5


# --------------------------------------------------
# Main loop
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(
        frame,
        1
    )

    landmarks = None


    # ----------------------------------------------
    # Try tracking first
    # ----------------------------------------------

    if tracker.is_tracking():

        landmarks = tracker.update(
            frame
        )


    # ----------------------------------------------
    # Periodic face detection
    # ----------------------------------------------

    frames_since_detection += 1

    should_redetect = (
        landmarks is None
        or frames_since_detection >= REDETECT_INTERVAL
    )


    if should_redetect:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )


        if len(faces) > 0:

            success, detected_landmarks = (
                facemark.fit(
                    frame,
                    faces
                )
            )


            if success:

                landmarks = (
                    detected_landmarks[0][0]
                )

                landmarks = (
                    tracker.correct(
                        landmarks
                    )
                )

                tracker.initialize(
                    frame,
                    landmarks
                )

                frames_since_detection = 0


    # ----------------------------------------------
    # Transform face
    # ----------------------------------------------

    if landmarks is not None:

        # ------------------------------------------
        # Eye expression
        # ------------------------------------------

        eye_data = get_eye_expression(
            landmarks
        )

        expression = (
            expression_controller.update(
                eye_data
            )
        )


        print(
            f"RAW "
            f"L:{eye_data['left_ear']:.3f} "
            f"R:{eye_data['right_ear']:.3f} | "
            f"SMOOTH "
            f"L:{expression['left_ear_smooth']:.3f} "
            f"R:{expression['right_ear_smooth']:.3f} | "
            f"BASE "
            f"L:{expression['left_baseline']:.3f} "
            f"R:{expression['right_baseline']:.3f} | "
            f"RATIO "
            f"L:{expression['left_ratio']:.2f} "
            f"R:{expression['right_ratio']:.2f} | "
            f"STATE "
            f"L:{expression['left_closed']} "
            f"R:{expression['right_closed']} | "
            f"START:{expression['blink_started']} "
            f"BLINK:{expression['blink_detected']} "
            f"COUNT:{expression['blink_count']}"
        )


        # ------------------------------------------
        # Warp target face
        # ------------------------------------------

        warped_face = warp_face(
            target_image,
            target_landmarks,
            frame.shape,
            landmarks
        )


        # ------------------------------------------
        # Blend
        # ------------------------------------------

        transformed = blend_face(
            frame,
            warped_face,
            landmarks
        )


        
        # ------------------------------------------
        # Blink and landmark diagnostics
        # ------------------------------------------

        debug_frame = transformed.copy()

        # Draw the two eye contours.
        eye_contours = [
            list(range(36, 42)),
            list(range(42, 48)),
        ]

        for contour in eye_contours:
            points = landmarks[contour].astype(
                "int32"
            ).reshape((-1, 1, 2))

            cv2.polylines(
                debug_frame,
                [points],
                True,
                (0, 255, 0),
                2,
            )

        # Draw individual landmarks.
        for i in range(36, 48):
            x, y = landmarks[i].astype(int)

            cv2.circle(
                debug_frame,
                (x, y),
                3,
                (0, 0, 255),
                -1,
            )

        # Show raw and smoothed measurements.
        diagnostic_lines = [
            (
                f"EAR L:{eye_data['left_ear']:.3f} "
                f"R:{eye_data['right_ear']:.3f}"
            ),
            (
                f"RATIO L:{expression['left_ratio']:.2f} "
                f"R:{expression['right_ratio']:.2f}"
            ),
            (
                f"CLOSED L:{expression['left_closed']} "
                f"R:{expression['right_closed']}"
            ),
            (
                f"BLINKS: {expression['blink_count']}"
            ),
        ]

        for index, line in enumerate(diagnostic_lines):
            cv2.putText(
                debug_frame,
                line,
                (10, 30 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if expression["blink_detected"]:
            cv2.putText(
                debug_frame,
                "BLINK CONFIRMED!",
                (10, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(
            "Face Transformer",
            debug_frame
        )



    else:

        cv2.imshow(
            "Face Transformer",
            frame
        )


    # ----------------------------------------------
    # Quit
    # ----------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()

cv2.destroyAllWindows()