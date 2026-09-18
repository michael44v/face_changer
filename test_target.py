import cv2

from src.transformation.target_face import TargetFace


facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("models/lbfmodel.yaml")


target = TargetFace(
    "assets/targets/image.png",
    facemark
)

target_image = target.get_image()
target_landmarks = target.get_landmarks()


for x, y in target_landmarks:
    cv2.circle(
        target_image,
        (int(x), int(y)),
        2,
        (0, 255, 0),
        -1
    )


cv2.imshow(
    "Target Face",
    target_image
)

cv2.waitKey(0)
cv2.destroyAllWindows()