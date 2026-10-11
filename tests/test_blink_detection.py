import pytest
import numpy as np

from src.expressions.eye_expression import _eye_aspect_ratio, get_eye_expression, get_facial_expressions
from src.expressions.expression_controller import ExpressionController


def make_eye_landmarks(horizontal=30.0, vertical1=10.0, vertical2=10.0):
    return np.array([
        [0.0, 0.0],
        [10.0, -vertical1],
        [20.0, -vertical2],
        [horizontal, 0.0],
        [20.0, 0.0],
        [10.0, 0.0]
    ], dtype=np.float64)


def test_ear_calculation():
    eye = make_eye_landmarks(horizontal=30.0, vertical1=9.0, vertical2=9.0)
    ear = _eye_aspect_ratio(eye)
    assert abs(ear - 0.30) < 1e-4


def test_ear_zero_division():
    eye = np.zeros((6, 2))
    assert _eye_aspect_ratio(eye) == 0.0


def test_blink_controller_single_blink():
    controller = ExpressionController()

    open_eye = {"left_ear": 0.30, "right_ear": 0.30, "valid": True}
    closed_eye = {"left_ear": 0.12, "right_ear": 0.12, "valid": True}

    # Initialize baseline over several frames
    for _ in range(10):
        res = controller.update(open_eye)

    assert abs(res["left_baseline"] - 0.30) < 0.02
    assert res["blink_count"] == 0
    assert res["left_closure"] == 0.0

    # Start blink (closed eye)
    res_close = controller.update(closed_eye)
    assert res_close["blink_started"] == True
    assert res_close["left_closed"] == True
    assert res_close["right_closed"] == True
    assert res_close["left_closure"] > 0.80

    # Reopen eye
    res_open = controller.update(open_eye)
    assert res_open["blink_detected"] == True
    assert res_open["blink_count"] == 1
    assert res_open["left_closed"] == False


def test_prolonged_closure_does_not_count_as_blink():
    controller = ExpressionController(max_blink_frames=5)
    open_eye = {"left_ear": 0.30, "right_ear": 0.30, "valid": True}
    closed_eye = {"left_ear": 0.10, "right_ear": 0.10, "valid": True}

    for _ in range(5):
        controller.update(open_eye)

    for _ in range(10):
        controller.update(closed_eye)

    res_reopen = controller.update(open_eye)
    assert res_reopen["blink_count"] == 0
