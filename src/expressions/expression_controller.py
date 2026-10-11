
class ExpressionController:
    def __init__(
        self,
        smoothing=0.35,
        baseline_learning=0.015,
        close_ratio=0.68,
        reopen_ratio=0.82,
        close_frames=2,
        reopen_frames=2,
        min_blink_frames=2,
        max_blink_frames=20,
    ):
        self.smoothing = smoothing
        self.baseline_learning = baseline_learning

        self.close_ratio = close_ratio
        self.reopen_ratio = reopen_ratio

        self.close_frames = close_frames
        self.reopen_frames = reopen_frames
        self.min_blink_frames = min_blink_frames
        self.max_blink_frames = max_blink_frames

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

    @staticmethod
    def _valid_ear(value):
        return (
            isinstance(value, (int, float))
            and 0.0 < value < 1.0
        )

    def _smooth(self, previous, current):
        if previous is None:
            return current

        return (
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
        """
        Update one eye using separate closing and reopening
        thresholds to reduce rapid state switching.
        """
        smooth = self._smooth(smooth, ear)

        if baseline is None:
            baseline = smooth

        ratio = smooth / max(baseline, 1e-6)

        if not closed:
            # Learn the normal open-eye EAR gradually.
            # Do not learn a closed eye as the new baseline.
            if ratio > self.reopen_ratio:
                baseline = (
                    (1.0 - self.baseline_learning) * baseline
                    + self.baseline_learning * smooth
                )

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

        return (
            smooth,
            baseline,
            ratio,
            closed,
            close_count,
            open_count,
        )

    def update(self, eye_data):
        # Events last for one update only.
        self.blink_started = False
        self.blink_detected = False

        left_ear = float(eye_data["left_ear"])
        right_ear = float(eye_data["right_ear"])

        if not (
            self._valid_ear(left_ear)
            and self._valid_ear(right_ear)
        ):
            return self._result(
                left_ratio=0.0,
                right_ratio=0.0,
            )

        (
            self.left_smooth,
            self.left_baseline,
            left_ratio,
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

        both_closed = (
            self.left_closed and self.right_closed
        )

        both_open = (
            not self.left_closed and not self.right_closed
        )

        if both_closed:
            self.both_closed_frames += 1

            if not self.blink_in_progress:
                self.blink_in_progress = True
                self.blink_started = True

        elif both_open:
            if self.blink_in_progress:
                duration = self.both_closed_frames

                if (
                    self.min_blink_frames
                    <= duration
                    <= self.max_blink_frames
                ):
                    self.blink_detected = True
                    self.blink_count += 1

                self.blink_in_progress = False

            self.both_closed_frames = 0

        else:
            # A prolonged asymmetric closure should not remain
            # classified as a normal blink.
            if (
                self.blink_in_progress
                and self.both_closed_frames
                > self.max_blink_frames
            ):
                self.blink_in_progress = False
                self.both_closed_frames = 0

        return self._result(left_ratio, right_ratio)

    def _result(self, left_ratio=0.0, right_ratio=0.0):
        return {
            "left_ear_smooth": (
                self.left_smooth
                if self.left_smooth is not None else 0.0
            ),
            "right_ear_smooth": (
                self.right_smooth
                if self.right_smooth is not None else 0.0
            ),
            "left_baseline": (
                self.left_baseline
                if self.left_baseline is not None else 0.0
            ),
            "right_baseline": (
                self.right_baseline
                if self.right_baseline is not None else 0.0
            ),
            "left_ratio": left_ratio,
            "right_ratio": right_ratio,
            "left_closed": self.left_closed,
            "right_closed": self.right_closed,
            "blink_started": self.blink_started,
            "blink_detected": self.blink_detected,
            "blink_count": self.blink_count,
        }
