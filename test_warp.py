import cv2

from src.transformation.target_face import TargetFace
from src.transformation.face_warper import warp_face


# -----------------------------
# Load target
# -----------------------------

facemark = cv2.face.createFacemarkLBF()

facemark.loadModel(
    "models/lbfmodel.yaml"
)

target = TargetFace(
    "assets/targets/image.png",
    facemark
)

target_image = target.get_image()
target_landmarks = target.get_landmarks()


# -----------------------------
# Create a test destination
# -----------------------------

destination = cv2.imread(
    "assets/targets/image.png"
)

destination_landmarks = target_landmarks.copy()


# -----------------------------
# Warp
# -----------------------------

result = warp_face(
    target_image,
    target_landmarks,
    destination,
    destination_landmarks
)


# -----------------------------
# Display
# -----------------------------

cv2.imshow(
    "Warp Test",
    result
)

cv2.waitKey(0)
cv2.destroyAllWindows()