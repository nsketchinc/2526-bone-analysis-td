"""Tests for CHOP-channel / dict / JSON serialization."""

from __future__ import annotations

import json

from td_pose import PoseResult, channel_names, result_to_channels, result_to_dict
from td_pose.formats import result_to_json
from td_pose.keypoints import NUM_KEYPOINTS


def test_channel_count():
    # 17 keypoints * (tx, ty, conf) + score + id + 4 bbox = 57
    assert len(channel_names()) == NUM_KEYPOINTS * 3 + 2 + 4 == 57


def test_channels_have_one_value_per_person(sample_pose_result):
    channels = result_to_channels(sample_pose_result)
    for name, values in channels.items():
        assert len(values) == sample_pose_result.num_people, name


def test_empty_result_gives_empty_channels():
    empty = PoseResult(persons=(), image_width=640, image_height=480)
    channels = result_to_channels(empty)
    assert set(channels) == set(channel_names())
    assert all(values == [] for values in channels.values())


def test_normalize_and_flip_y(sample_pose_result):
    # First person nose at pixel (320, 240) in a 640x480 frame.
    channels = result_to_channels(sample_pose_result, normalize=True, flip_y=True)
    assert channels["nose:tx"][0] == 320.0 / 640.0
    # flip: y_out = 1 - (240/480) = 0.5
    assert channels["nose:ty"][0] == 1.0 - (240.0 / 480.0)


def test_no_normalize_no_flip(sample_pose_result):
    channels = result_to_channels(sample_pose_result, normalize=False, flip_y=False)
    assert channels["nose:tx"][0] == 320.0
    assert channels["nose:ty"][0] == 240.0


def test_id_defaults_to_minus_one_when_untracked(sample_pose_result):
    channels = result_to_channels(sample_pose_result)
    assert channels["id"][0] == 5.0  # tracked
    assert channels["id"][1] == -1.0  # untracked person


def test_result_to_dict_structure(sample_pose_result):
    data = result_to_dict(sample_pose_result)
    assert data["num_people"] == 2
    assert data["width"] == 640 and data["height"] == 480
    assert len(data["people"]) == 2
    person = data["people"][0]
    assert len(person["keypoints"]) == NUM_KEYPOINTS
    assert person["id"] == 5


def test_result_to_json_roundtrips(sample_pose_result):
    text = result_to_json(sample_pose_result)
    parsed = json.loads(text)
    assert parsed["num_people"] == 2
