"""Tests for EngineConfig validation."""

from __future__ import annotations

import pytest

from td_pose import EngineConfig
from td_pose.config import DEFAULT_MODEL_PATH


def test_defaults_are_valid():
    cfg = EngineConfig()
    assert cfg.model_path == DEFAULT_MODEL_PATH
    assert 0.0 <= cfg.confidence <= 1.0
    assert cfg.imgsz > 0
    assert cfg.max_det > 0


@pytest.mark.parametrize("bad_conf", [-0.1, 1.1, 2.0])
def test_invalid_confidence_raises(bad_conf):
    with pytest.raises(ValueError):
        EngineConfig(confidence=bad_conf)


def test_invalid_imgsz_raises():
    with pytest.raises(ValueError):
        EngineConfig(imgsz=0)


def test_empty_model_path_raises():
    with pytest.raises(ValueError):
        EngineConfig(model_path="")


def test_with_model_returns_new_config():
    cfg = EngineConfig()
    other = cfg.with_model("yolo11n-pose.pt")
    assert other.model_path == "yolo11n-pose.pt"
    assert cfg.model_path == DEFAULT_MODEL_PATH  # original unchanged (immutable)
