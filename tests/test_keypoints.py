"""Tests for the COCO keypoint / skeleton topology."""

from __future__ import annotations

from td_pose import keypoints as kp


def test_keypoint_count_is_seventeen():
    assert kp.NUM_KEYPOINTS == 17
    assert len(kp.COCO_KEYPOINT_NAMES) == 17


def test_skeleton_indices_are_in_range():
    for a, b in kp.COCO_SKELETON:
        assert 0 <= a < kp.NUM_KEYPOINTS
        assert 0 <= b < kp.NUM_KEYPOINTS
        assert a != b


def test_bone_names_align_with_skeleton():
    names = kp.bone_names()
    assert len(names) == kp.NUM_BONES == len(kp.COCO_SKELETON)
    # Names are derived from the keypoint names of each edge.
    first_a, first_b = kp.COCO_SKELETON[0]
    expected = f"{kp.COCO_KEYPOINT_NAMES[first_a]}__{kp.COCO_KEYPOINT_NAMES[first_b]}"
    assert names[0] == expected


def test_bone_colors_one_per_bone():
    colors = kp.bone_colors()
    assert len(colors) == kp.NUM_BONES
    for rgb in colors:
        assert len(rgb) == 3
        assert all(0 <= c <= 255 for c in rgb)
