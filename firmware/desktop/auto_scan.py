"""Desktop mirror of the portable C++ Smart Sort YOLO gating policy.

The 500 ms period starts with the first eligible YOLO detection, not object entry.
No background-reference photograph is needed. This is *not* hand detection.
"""
from dataclasses import dataclass
from enum import Enum
from math import hypot


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class State(str, Enum):
    SEARCHING = "SEARCHING"
    CONFIRMING = "CONFIRMING"
    CAPTURE_PENDING = "CAPTURE_PENDING"
    WAIT_FOR_REMOVAL = "WAIT_FOR_REMOVAL"


class AutoScan:
    def __init__(self, confirm_ms=500, rearm_ms=1000, threshold=0.55,
                 min_area=0.02, zone=(0.20, 0.15, 0.80, 0.85), max_motion=0.20):
        self.confirm_ms = confirm_ms
        self.rearm_ms = rearm_ms
        self.threshold = threshold
        self.min_area = min_area
        self.zone = zone
        self.max_motion = max_motion
        self.reset()

    def reset(self):
        self.state = State.SEARCHING
        self.target = None
        self.first_seen_ms = None
        self.absence_since_ms = None

    def eligible(self, detection):
        d = detection
        if d.class_id <= 0 or d.confidence < self.threshold:
            return False  # COCO 0 is person, not hand
        if not (0 <= d.x1 < d.x2 <= 1 and 0 <= d.y1 < d.y2 <= 1):
            return False
        cx, cy = d.center
        left, top, right, bottom = self.zone
        return (d.x2 - d.x1) * (d.y2 - d.y1) >= self.min_area and (left <= cx <= right) and (top <= cy <= bottom)

    def same_item(self, a, b):
        return a.class_id == b.class_id and hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]) <= self.max_motion

    def update(self, detections, now_ms):
        """Returns True exactly once when the device should take a photo."""
        eligible = [d for d in detections if self.eligible(d)]

        if self.state == State.CAPTURE_PENDING:
            return False
        if self.state == State.WAIT_FOR_REMOVAL:
            if eligible:
                self.absence_since_ms = None
            elif self.absence_since_ms is None:
                self.absence_since_ms = now_ms
            elif now_ms - self.absence_since_ms >= self.rearm_ms:
                self.reset()
            return False

        if self.state == State.SEARCHING:
            if eligible:
                self.target = max(eligible, key=lambda d: d.confidence)
                self.first_seen_ms = now_ms
                self.state = State.CONFIRMING
            return False

        matches = [d for d in eligible if self.same_item(self.target, d)]
        if not matches:
            self.state = State.SEARCHING
            if eligible:
                self.target = max(eligible, key=lambda d: d.confidence)
                self.first_seen_ms = now_ms
                self.state = State.CONFIRMING
            return False

        self.target = max(matches, key=lambda d: d.confidence)
        if now_ms - self.first_seen_ms >= self.confirm_ms:
            self.state = State.CAPTURE_PENDING
            return True
        return False

    def capture_finished(self):
        if self.state == State.CAPTURE_PENDING:
            self.state = State.WAIT_FOR_REMOVAL
            self.absence_since_ms = None
