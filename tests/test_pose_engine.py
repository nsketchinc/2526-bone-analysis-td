"""Tests for PoseEngine parsing and inference glue (model mocked)."""

from __future__ import annotations

import numpy as np
import pytest

from td_pose import EngineConfig, PoseEngine
from td_pose.pose_engine import PoseEngine as Engine

from conftest import FakeKeypoints, FakeModel, FakeResult, make_xy


def _frame(width: int = 64, height: int = 48) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_parse_result_extracts_people(two_person_result):
    persons = Engine._parse_result(two_person_result)
    assert len(persons) == 2
    assert persons[0].keypoints_xy.shape == (17, 2)
    assert persons[0].score == pytest.approx(0.8)
    assert persons[0].track_id == 11
    assert persons[1].track_id == 22
    assert persons[0].bbox_xyxy == (10.0, 20.0, 110.0, 220.0)


def test_parse_result_without_boxes_uses_defaults():
    xy = make_xy(1, 17)
    result = FakeResult(FakeKeypoints(xy, None), boxes=None)
    persons = Engine._parse_result(result)
    assert len(persons) == 1
    assert persons[0].score == 1.0  # default confidence
    assert persons[0].track_id is None
    assert persons[0].bbox_xyxy == (0.0, 0.0, 0.0, 0.0)
    # Missing keypoint conf defaults to ones.
    assert np.allclose(persons[0].keypoints_conf, 1.0)


def test_parse_result_skips_wrong_topology():
    xy = make_xy(1, 5)  # not 17 keypoints
    result = FakeResult(FakeKeypoints(xy, None), boxes=None)
    assert Engine._parse_result(result) == ()


def test_parse_result_empty_keypoints():
    empty = np.empty((0, 17, 2), dtype=np.float32)
    result = FakeResult(FakeKeypoints(empty, None), boxes=None)
    assert Engine._parse_result(result) == ()


def test_infer_returns_dimensions_and_calls_predict(two_person_result):
    engine = PoseEngine(EngineConfig(device="cpu"))
    engine._model = FakeModel(two_person_result)  # bypass load()
    frame = _frame(64, 48)

    result = engine.infer(frame)

    assert result.image_width == 64
    assert result.image_height == 48
    assert result.num_people == 2
    assert len(engine._model.predict_calls) == 1
    # RGB -> BGR conversion is applied before handing to the model.
    sent = engine._model.predict_calls[0]["source"]
    assert np.array_equal(sent, frame[:, :, ::-1])


def test_infer_tracking_uses_track(two_person_result):
    engine = PoseEngine(EngineConfig(device="cpu", enable_tracking=True))
    engine._model = FakeModel(two_person_result)
    engine.infer(_frame())
    assert len(engine._model.track_calls) == 1
    assert engine._model.track_calls[0]["persist"] is True


def test_infer_rejects_bad_shape():
    engine = PoseEngine(EngineConfig(device="cpu"))
    engine._model = FakeModel(FakeResult(FakeKeypoints(make_xy(1), None), None))
    with pytest.raises(ValueError):
        engine.infer(np.zeros((10, 10), dtype=np.uint8))  # missing channel axis


def test_load_falls_back_when_primary_model_missing(monkeypatch):
    """Primary model raises, first fallback succeeds."""
    attempted: list[str] = []

    class FlakyYOLO:
        def __init__(self, path):
            attempted.append(path)
            if path == "yolo26n-pose.pt":
                raise FileNotFoundError("no yolo26 in this ultralytics build")

    fake_module = type("M", (), {"YOLO": FlakyYOLO})
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", fake_module)

    engine = PoseEngine(EngineConfig())
    loaded = engine.load()
    assert loaded == "yolo11n-pose.pt"
    assert attempted[0] == "yolo26n-pose.pt"
