from time import monotonic
from typing import Optional, Sequence, Tuple
import numpy as np

class WristFlipDetector:

    def __init__ (
        self,
        smoothing: float = 0.25,
        enter_threshold: float = 0.35,
        stable_frames: int = 4,
        cooldown: float = 0.5,
        max_missing_frames: int = 10,
        denominator_eps: float = 0.0001,
    ) -> None:

        if not 0.0 < smoothing <= 1.0:
            raise ValueError ("smoothing must be in the range (0, 1]")
        if not 0.0 < enter_threshold < 1.0:
            raise ValueError ("enter_threshold must be in the range (0, 1)")
        if stable_frames < 1:
            raise ValueError ("stable_frames must be at least 1")
        if cooldown < 0.0:
            raise ValueError ("cooldown cannot be negative")
        if max_missing_frames < 1:
            raise ValueError ("max_missing_frames must be at least 1")
        if denominator_eps <= 0.0:
            raise ValueError ("denominator_eps must be positive")

        self.smoothing = smoothing
        self.enter_threshold = enter_threshold
        self.stable_frames = stable_frames
        self.cooldown = cooldown
        self.max_missing_frames = max_missing_frames
        self.denominator_eps = denominator_eps

        self.filtered_score: Optional[float] = None
        self.stable_side: Optional[int] = None
        self.candidate_side: Optional[int] = None
        self.candidate_frames = 0
        self.missing_frames = 0
        self.last_event_time = - float ("inf")

    def reset (self) -> None:
        self.filtered_score = None
        self.stable_side = None
        self.candidate_side = None
        self.candidate_frames = 0
        self.missing_frames = 0

    def orientation_score (
        self,
        landmarks: Sequence[Sequence[float]]
    ) -> Optional[float]:
        points = np.asarray (landmarks, dtype = np.float32)
        if points.shape != (21, 2):
            return None

        wrist = points [0]
        index_mcp = points [5]
        pinky_mcp = points [17]
        index_vector = index_mcp - wrist
        pinky_vector = pinky_mcp - wrist

        denominator = float ( np.linalg.norm (index_vector) * np.linalg.norm (pinky_vector) )
        if denominator < self.denominator_eps:
            return None

        cross = float ( index_vector [0] * pinky_vector [1] - index_vector [1] * pinky_vector [0] )
        return cross / denominator

    def update (
        self,
        landmarks: Optional[Sequence[Sequence[float]]],
        now: Optional[float] = None
    ) -> Tuple[bool, str, Optional[float]]:

        if now is None:
            current_time = monotonic ()
        else:
            current_time = now

        # No hand detected.
        if landmarks is None:
            self.missing_frames += 1
            if self.missing_frames >= self.max_missing_frames:
                self.reset ()
            return False, "NO_HAND", self.filtered_score

        raw_score = self.orientation_score (landmarks)
        if raw_score is None:
            return False, "INVALID", self.filtered_score

        self.missing_frames = 0
        if self.filtered_score is None:
            self.filtered_score = raw_score
        else:
            self.filtered_score = (
                self.smoothing * raw_score
                + (1.0 - self.smoothing) * self.filtered_score
            )

        if self.filtered_score > self.enter_threshold:
            current_side = 1
        elif self.filtered_score < - self.enter_threshold:
            current_side = - 1
        else:
            current_side = 0

        if current_side == 0:
            self.candidate_side = None
            self.candidate_frames = 0
            return False, "TRANSITION", self.filtered_score
        if current_side == self.candidate_side:
            self.candidate_frames += 1
        else:
            self.candidate_side = current_side
            self.candidate_frames = 1

        flip_detected = False
        if self.candidate_frames >= self.stable_frames:
            side_changed = ( self.stable_side is not None and current_side != self.stable_side )
            outside_cooldown = ( current_time - self.last_event_time >= self.cooldown )

            if side_changed and outside_cooldown:
                flip_detected = True
                self.last_event_time = current_time

            self.stable_side = current_side

        if self.stable_side is None:
            state = "WAITING"
        elif self.stable_side > 0:
            state = "SIDE_A"
        else:
            state = "SIDE_B"

        return flip_detected, state, self.filtered_score
