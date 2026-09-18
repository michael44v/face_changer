import numpy as np


class ExpressionController:

    def __init__(self):

        # -----------------------------------------
        # EAR baselines
        # -----------------------------------------

        self.left_baseline = None
        self.right_baseline = None

        # Slowly adapt to the user's natural eye shape
        self.baseline_alpha = 0.01

        # -----------------------------------------
        # Smoothed EAR
        # -----------------------------------------

        self.left_ear_smooth = None
        self.right_ear_smooth = None

        self.ear_smoothing = 0.45

        # -----------------------------------------
        # Eye states
        # -----------------------------------------

        self.left_closed = False
        self.right_closed = False

        # Consecutive frame counters
        self.left_close_frames = 0
        self.right_close_frames = 0

        self.left_open_frames = 0
        self.right_open_frames = 0

        # -----------------------------------------
        # Thresholds
        # -----------------------------------------

        # Relative to user's normal open eye.
        self.close_ratio = 0.72
        self.open_ratio = 0.82

        # Absolute safety threshold.
        self.absolute_close_threshold = 0.24

        # Require several frames before declaring
        # an eye actually closed.
        self.required_close_frames = 2
        self.required_open_frames = 2

        # -----------------------------------------
        # Blink event
        # -----------------------------------------

        self.blink_active = False

    # =================================================
    # BASELINE
    # =================================================

    def _update_baseline(
        self,
        ear,
        baseline
    ):

        if baseline is None:
            return ear

        # Do not allow a closing eye to drag
        # the baseline downward.
        if ear > baseline * 0.90:

            baseline = (
                (1.0 - self.baseline_alpha)
                * baseline
                +
                self.baseline_alpha
                * ear
            )

        return baseline

    # =================================================
    # SMOOTH EAR
    # =================================================

    def _smooth_ear(
        self,
        current,
        previous
    ):

        if previous is None:
            return current

        return (
            self.ear_smoothing * previous
            +
            (1.0 - self.ear_smoothing) * current
        )

    # =================================================
    # UPDATE
    # =================================================

    def update(self, eye_data):

        left_ear = float(
            eye_data["left_ear"]
        )

        right_ear = float(
            eye_data["right_ear"]
        )

        # -----------------------------------------
        # Smooth EAR
        # -----------------------------------------

        self.left_ear_smooth = self._smooth_ear(
            left_ear,
            self.left_ear_smooth
        )

        self.right_ear_smooth = self._smooth_ear(
            right_ear,
            self.right_ear_smooth
        )

        left_ear = self.left_ear_smooth
        right_ear = self.right_ear_smooth

        # -----------------------------------------
        # Update individual baselines
        # -----------------------------------------

        self.left_baseline = self._update_baseline(
            left_ear,
            self.left_baseline
        )

        self.right_baseline = self._update_baseline(
            right_ear,
            self.right_baseline
        )

        # -----------------------------------------
        # Relative openness
        # -----------------------------------------

        left_ratio = (
            left_ear
            / max(self.left_baseline, 0.001)
        )

        right_ratio = (
            right_ear
            / max(self.right_baseline, 0.001)
        )

        # -----------------------------------------
        # Closing conditions
        #
        # Use BOTH relative and absolute EAR.
        # -----------------------------------------

        left_is_closing = (
            left_ratio < self.close_ratio
            or
            left_ear < self.absolute_close_threshold
        )

        right_is_closing = (
            right_ratio < self.close_ratio
            or
            right_ear < self.absolute_close_threshold
        )

        # -----------------------------------------
        # LEFT EYE
        # -----------------------------------------

        if left_is_closing:

            self.left_close_frames += 1
            self.left_open_frames = 0

        else:

            self.left_open_frames += 1
            self.left_close_frames = 0

        # -----------------------------------------
        # RIGHT EYE
        # -----------------------------------------

        if right_is_closing:

            self.right_close_frames += 1
            self.right_open_frames = 0

        else:

            self.right_open_frames += 1
            self.right_close_frames = 0

        # -----------------------------------------
        # Close left eye
        # -----------------------------------------

        if (
            not self.left_closed
            and
            self.left_close_frames
            >= self.required_close_frames
        ):

            self.left_closed = True

        # -----------------------------------------
        # Close right eye
        # -----------------------------------------

        if (
            not self.right_closed
            and
            self.right_close_frames
            >= self.required_close_frames
        ):

            self.right_closed = True

        # -----------------------------------------
        # Re-open left eye
        # -----------------------------------------

        left_has_recovered = (
            left_ratio > self.open_ratio
            and
            left_ear > self.absolute_close_threshold
        )

        if (
            self.left_closed
            and
            self.left_open_frames
            >= self.required_open_frames
            and
            left_has_recovered
        ):

            self.left_closed = False

        # -----------------------------------------
        # Re-open right eye
        # -----------------------------------------

        right_has_recovered = (
            right_ratio > self.open_ratio
            and
            right_ear > self.absolute_close_threshold
        )

        if (
            self.right_closed
            and
            self.right_open_frames
            >= self.required_open_frames
            and
            right_has_recovered
        ):

            self.right_closed = False

        # -----------------------------------------
        # Overall blink
        # -----------------------------------------

        blink_now = (
            self.left_closed
            or
            self.right_closed
        )

        # Only true on transition:
        #
        # False -> True
        #

        blink_started = (
            blink_now
            and
            not self.blink_active
        )

        self.blink_active = blink_now

        # -----------------------------------------
        # Normalized eye openness
        # -----------------------------------------

        left_open = np.clip(
            left_ratio,
            0.0,
            1.0
        )

        right_open = np.clip(
            right_ratio,
            0.0,
            1.0
        )

        # -----------------------------------------
        # Return
        # -----------------------------------------

        return {

            "left_open": float(
                left_open
            ),

            "right_open": float(
                right_open
            ),

            "left_ear_smooth": float(
                self.left_ear_smooth
            ),

            "right_ear_smooth": float(
                self.right_ear_smooth
            ),

            "left_closed": self.left_closed,

            "right_closed": self.right_closed,

            "blink": blink_now,

            "blink_started": blink_started,

            "left_baseline": float(
                self.left_baseline
            ),

            "right_baseline": float(
                self.right_baseline
            ),

            "left_ratio": float(
                left_ratio
            ),

            "right_ratio": float(
                right_ratio
            )
        }