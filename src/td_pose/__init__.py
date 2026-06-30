"""td_pose — real-time YOLO26 pose ("bone") analysis engine for TouchDesigner.

The package is intentionally decoupled from TouchDesigner: it only depends on
``numpy`` (always) and ``ultralytics`` (lazily, at inference time). This lets the
core be unit-tested off-TD while the ``td/`` glue scripts import it inside TD.
"""

from __future__ import annotations

from .config import EngineConfig
from .draw import draw_pose
from .formats import (
    channel_names,
    result_to_channels,
    result_to_dict,
    result_to_json,
)
from .keypoints import (
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON,
    NUM_BONES,
    NUM_KEYPOINTS,
    bone_colors,
    bone_names,
)
from .pose_engine import Person, PoseEngine, PoseResult

__all__ = [
    "EngineConfig",
    "PoseEngine",
    "Person",
    "PoseResult",
    "draw_pose",
    "channel_names",
    "result_to_channels",
    "result_to_dict",
    "result_to_json",
    "COCO_KEYPOINT_NAMES",
    "COCO_SKELETON",
    "NUM_KEYPOINTS",
    "NUM_BONES",
    "bone_names",
    "bone_colors",
]

__version__ = "0.1.0"
