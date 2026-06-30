"""Shared fixtures: fake Ultralytics result/model objects (no torch needed)."""

from __future__ import annotations

import numpy as np
import pytest

from td_pose import Person, PoseResult


class FakeKeypoints:
    def __init__(self, xy: np.ndarray, conf: np.ndarray | None) -> None:
        self.xy = xy
        self.conf = conf


class FakeBoxes:
    def __init__(self, conf=None, ids=None, xyxy=None) -> None:
        self.conf = conf
        self.id = ids
        self.xyxy = xyxy


class FakeResult:
    def __init__(self, keypoints, boxes) -> None:
        self.keypoints = keypoints
        self.boxes = boxes


class FakeModel:
    """Stands in for ``ultralytics.YOLO`` — records calls, returns a fixed result."""

    def __init__(self, result) -> None:
        self.result = result
        self.predict_calls: list[dict] = []
        self.track_calls: list[dict] = []

    def predict(self, source, **kwargs):
        self.predict_calls.append({"source": source, **kwargs})
        return [self.result]

    def track(self, source, **kwargs):
        self.track_calls.append({"source": source, **kwargs})
        return [self.result]


def make_xy(num_people: int = 2, num_kp: int = 17) -> np.ndarray:
    """Deterministic keypoint pixel coordinates, shape (n, K, 2)."""
    base = np.arange(num_people * num_kp * 2, dtype=np.float32)
    return base.reshape(num_people, num_kp, 2) + 1.0


@pytest.fixture
def two_person_result():
    xy = make_xy(2, 17)
    conf = np.full((2, 17), 0.9, dtype=np.float32)
    boxes = FakeBoxes(
        conf=np.array([0.8, 0.7], dtype=np.float32),
        ids=np.array([11, 22], dtype=np.float32),
        xyxy=np.array([[10, 20, 110, 220], [30, 40, 130, 240]], dtype=np.float32),
    )
    return FakeResult(FakeKeypoints(xy, conf), boxes)


@pytest.fixture
def sample_pose_result():
    """A real PoseResult with two people in a 640x480 frame."""
    persons = (
        Person(
            keypoints_xy=np.array([[320.0, 240.0]] * 17, dtype=np.float32),
            keypoints_conf=np.full(17, 0.9, dtype=np.float32),
            bbox_xyxy=(100.0, 100.0, 200.0, 300.0),
            score=0.8,
            track_id=5,
        ),
        Person(
            keypoints_xy=np.array([[160.0, 120.0]] * 17, dtype=np.float32),
            keypoints_conf=np.full(17, 0.5, dtype=np.float32),
            bbox_xyxy=(0.0, 0.0, 50.0, 50.0),
            score=0.6,
            track_id=None,
        ),
    )
    return PoseResult(persons=persons, image_width=640, image_height=480)
