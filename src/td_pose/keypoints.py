"""COCO 17-keypoint pose topology used by Ultralytics YOLO pose models.

The keypoint order and the skeleton (bone) edges below match exactly what
Ultralytics emits for ``yolo26n-pose.pt`` / ``yolo11n-pose.pt`` /
``yolov8n-pose.pt`` so downstream consumers can rely on stable indices.
"""

from __future__ import annotations

# Canonical COCO keypoint order (index == channel index emitted by YOLO pose).
COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

NUM_KEYPOINTS: int = len(COCO_KEYPOINT_NAMES)

# Skeleton "bones": pairs of keypoint indices that are connected.
# This is the 0-indexed equivalent of the Ultralytics default COCO skeleton.
COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (15, 13),  # left ankle  - left knee
    (13, 11),  # left knee   - left hip
    (16, 14),  # right ankle - right knee
    (14, 12),  # right knee  - right hip
    (11, 12),  # left hip    - right hip
    (5, 11),   # left shoulder  - left hip
    (6, 12),   # right shoulder - right hip
    (5, 6),    # left shoulder  - right shoulder
    (5, 7),    # left shoulder  - left elbow
    (6, 8),    # right shoulder - right elbow
    (7, 9),    # left elbow  - left wrist
    (8, 10),   # right elbow - right wrist
    (1, 2),    # left eye - right eye
    (0, 1),    # nose - left eye
    (0, 2),    # nose - right eye
    (1, 3),    # left eye  - left ear
    (2, 4),    # right eye - right ear
    (3, 5),    # left ear  - left shoulder
    (4, 6),    # right ear - right shoulder
)

NUM_BONES: int = len(COCO_SKELETON)


def bone_names() -> tuple[str, ...]:
    """Human-readable name for each bone, e.g. ``left_ankle__left_knee``."""
    return tuple(
        f"{COCO_KEYPOINT_NAMES[a]}__{COCO_KEYPOINT_NAMES[b]}"
        for a, b in COCO_SKELETON
    )


# RGB colors (0-255) per bone, for overlay rendering. Left side warm, right cool.
_LEFT = (0, 200, 255)
_RIGHT = (255, 120, 0)
_CENTER = (0, 255, 120)


def bone_colors() -> tuple[tuple[int, int, int], ...]:
    """A fixed RGB color per bone to make left/right limbs distinguishable."""
    colors: list[tuple[int, int, int]] = []
    for a, b in COCO_SKELETON:
        name_a = COCO_KEYPOINT_NAMES[a]
        name_b = COCO_KEYPOINT_NAMES[b]
        if name_a.startswith("left") or name_b.startswith("left"):
            colors.append(_LEFT)
        elif name_a.startswith("right") or name_b.startswith("right"):
            colors.append(_RIGHT)
        else:
            colors.append(_CENTER)
    return tuple(colors)
