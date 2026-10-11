import numpy as np


class ExpressionController:
    """
    Tracks and smoothes facial expressions, maintains adaptive open-eye baselines,
    computes continuous eye-closure ratios, and implements a state machine for blink detection.
    """

    def __init__(
        self,
        smoothing=0.50,
        baseline_learning=0.02,
        close_ratio=0.78,
        reopen_ratio=0.82,
        close_frames=1,
        reopen_frames=1,
        min_blink_frames=1,
        max_blink_frames=25,
        min_baseline=0.18,
        max_baseline=0.42,
    ):
        self.smoothing = smoothing
        self.baseline_learning = baseline_learning

        self.close_ratio = close_ratio
        self.reopen_ratio = reopen_ratio

        self.close_frames = close_frames
        self.reopen_frames = reopen_frames
        self.min_blink_frames = min_blink_frames
        self.max_blink_frames = max_blink_frames

        self.min_baseline = min_baseline
        self.max_baseline = max_baseline

        self.left_smooth = None
        self.right_smooth = None

        self.left_baseline = None
        self.right_baseline = None

        self.left_closed = False
        self.right_closed = False

        self.left_close_count = 0
        self.right_close_count = 0

        self.left_open_count = 0
        self.right_open_count = 0

        self.both_closed_frames = 0
        self.blink_in_progress = False

        self.blink_started = False
        self.blink_detected = False
        self.blink_count = 0

        # Continuous mouth and eyebrow measurements
        self.mouth_open_smooth = 0.0
        self.smile_smooth = 0.0
        self.eyebrow_elevation_smooth = 0.0

    def reset(self):
        """Reset temporal state when tracking is lost."""
        self.left_smooth = None
        self.right_smooth = None
        self.left_baseline = None
        self.right_baseline = None
        self.left_closed = False
        self.right_closed = False
        self.left_close_count = 0
        self.right_close_count = 0
        self.left_open_count = 0
        self.right_open_count = 0
        self.both_closed_frames = 0
        self.blink_in_progress = False
        self.blink_started = False
        self.blink_detected = False

    @staticmethod
    def _valid_ear(value):
        return (
            isinstance(value, (int, float, np.number))
            and np.isfinite(value)
            and 0.02 < float(value) < 0.95
        )

    def _smooth(self, previous, current):
        if previous is None:
            return float(current)

        return float(
            self.smoothing * current
            + (1.0 - self.smoothing) * previous
        )

    def _update_eye(
        self,
        ear,
        smooth,
        baseline,
        closed,
        close_count,
        open_count,
    ):
        smooth = self._smooth(smooth, ear)

        if baseline is None:
            # Initialize baseline within sensible eye aspect ratio range
            baseline = float(np.clip(smooth, self.min_baseline, self.max_baseline))

        ratio = smooth / max(baseline, 1e-6)

        if not closed:
            # Learn normal open-eye baseline gradually ONLY when eye is open
            if ratio > self.reopen_ratio:
                baseline = (
                    (1.0 - self.baseline_learning) * baseline
                    + self.baseline_learning * smooth
                )
                baseline = float(np.clip(baseline, self.min_baseline, self.max_baseline))

            ratio = smooth / max(baseline, 1e-6)

            if ratio < self.close_ratio:
                close_count += 1
            else:
                close_count = 0

            if close_count >= self.close_frames:
                closed = True
                close_count = 0
                open_count = 0

        else:
            if ratio > self.reopen_ratio:
                open_count += 1
            else:
                open_count = 0

            if open_count >= self.reopen_frames:
                closed = False
                open_count = 0
                close_count = 0

        ratio = smooth / max(baseline, 1e-6)

        # Map ratio to continuous closure progress: 0.0 (fully open) -> 1.0 (fully closed)
        closure = float(np.clip((1.0 - ratio) / (1.0 - self.close_ratio + 1e-6), 0.0, 1.0))

        return (
            smooth,
            baseline,
            ratio,
            closure,
            closed,
            close_count,
            open_count,
        )

    def update(self, eye_data):
        self.blink_started = False
        self.blink_detected = False

        if eye_data is None or not eye_data.get("valid", True):
            return self._result(
                left_ratio=1.0,
                right_ratio=1.0,
                left_closure=0.0,
                right_closure=0.0,
            )

        left_ear = float(eye_data.get("left_ear", 0.0))
        right_ear = float(eye_data.get("right_ear", 0.0))

        if not (self._valid_ear(left_ear) and self._valid_ear(right_ear)):
            return self._result(
                left_ratio=1.0,
                right_ratio=1.0,
                left_closure=0.0,
                right_closure=0.0,
            )

        (
            self.left_smooth,
            self.left_baseline,
            left_ratio,
            left_closure,
            self.left_closed,
            self.left_close_count,
            self.left_open_count,
        ) = self._update_eye(
            left_ear,
            self.left_smooth,
            self.left_baseline,
            self.left_closed,
            self.left_close_count,
            self.left_open_count,
        )

        (
            self.right_smooth,
            self.right_baseline,
            right_ratio,
            right_closure,
            self.right_closed,
            self.right_close_count,
            self.right_open_count,
        ) = self._update_eye(
            right_ear,
            self.right_smooth,
            self.right_baseline,
            self.right_closed,
            self.right_close_count,
            self.right_open_count,
        )

        # Smooth additional facial features if present
        if "mouth_open" in eye_data:
            self.mouth_open_smooth = self._smooth(self.mouth_open_smooth, eye_data["mouth_open"])
        if "smile" in eye_data:
            self.smile_smooth = self._smooth(self.smile_smooth, eye_data["smile"])
        if "eyebrow_elevation" in eye_data:
            self.eyebrow_elevation_smooth = self._smooth(self.eyebrow_elevation_smooth, eye_data["eyebrow_elevation"])

        both_closed = self.left_closed and self.right_closed
        both_open = not self.left_closed and not self.right_closed

        if both_closed:
            self.both_closed_frames += 1

            if not self.blink_in_progress:
                self.blink_in_progress = True
                self.blink_started = True

        elif both_open:
            if self.blink_in_progress:
                duration = self.both_closed_frames

                if self.min_blink_frames <= duration <= self.max_blink_frames:
                    self.blink_detected = True
                    self.blink_count += 1

                self.blink_in_progress = False

            self.both_closed_frames = 0

        else:
            # Handle prolonged asymmetric closure
            if self.blink_in_progress and self.both_closed_frames > self.max_blink_frames:
                self.blink_in_progress = False
                self.both_closed_frames = 0

        return self._result(
            left_ratio=left_ratio,
            right_ratio=right_ratio,
            left_closure=left_closure,
            right_closure=right_closure,
        )

    def _result(
        self,
        left_ratio=1.0,
        right_ratio=1.0,
        left_closure=0.0,
        right_closure=0.0,
    ):
        return {
            "left_ear_smooth": self.left_smooth if self.left_smooth is not None else 0.0,
            "right_ear_smooth": self.right_smooth if self.right_smooth is not None else 0.0,
            "left_baseline": self.left_baseline if self.left_baseline is not None else 0.0,
            "right_baseline": self.right_baseline if self.right_baseline is not None else 0.0,
            "left_ratio": left_ratio,
            "right_ratio": right_ratio,
            "left_closure": left_closure,
            "right_closure": right_closure,
            "left_closed": self.left_closed,
            "right_closed": self.right_closed,
            "blink_started": self.blink_started,
            "blink_detected": self.blink_detected,
            "blink_count": self.blink_count,
            "mouth_open": self.mouth_open_smooth,
            "smile": self.smile_smooth,
            "eyebrow_elevation": self.eyebrow_elevation_smooth,
        }
