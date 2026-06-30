"""Tests for the skeleton overlay renderer (both cv2 and numpy paths)."""

from __future__ import annotations

import numpy as np

from td_pose import draw_pose
from td_pose import draw as draw_module


def _blank(width: int = 200, height: int = 200) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_draw_preserves_shape_and_dtype(sample_pose_result):
    frame = _blank()
    out = draw_pose(frame, sample_pose_result)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_draw_does_not_mutate_input(sample_pose_result):
    frame = _blank()
    original = frame.copy()
    draw_pose(frame, sample_pose_result)
    assert np.array_equal(frame, original)


def test_draw_changes_pixels_for_visible_person(sample_pose_result):
    frame = _blank()
    out = draw_pose(frame, sample_pose_result, min_conf=0.3)
    assert out.sum() > 0  # joints/bones were drawn


def test_low_confidence_keypoints_are_skipped(sample_pose_result):
    frame = _blank()
    # Both people are below this threshold -> nothing drawn.
    out = draw_pose(frame, sample_pose_result, min_conf=0.99)
    assert out.sum() == 0


def test_numpy_fallback_path(monkeypatch, sample_pose_result):
    """Force the cv2-less code path and confirm it still draws."""
    monkeypatch.setattr(draw_module, "_HAS_CV2", False)
    frame = _blank()
    out = draw_pose(frame, sample_pose_result, min_conf=0.3)
    assert out.sum() > 0
