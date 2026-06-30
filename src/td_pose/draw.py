"""Render the detected skeleton (bones + joints) onto an RGB frame.

Uses OpenCV when available (it ships transitively with Ultralytics). Falls back
to a pure-NumPy line rasterizer so the overlay still works in a TouchDesigner
Python that lacks ``cv2``.

All inputs/outputs use top-left origin, RGB, uint8.
"""

from __future__ import annotations

import numpy as np

from .keypoints import COCO_SKELETON, bone_colors
from .pose_engine import PoseResult

try:  # pragma: no cover - presence depends on environment
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore
    _HAS_CV2 = False

_JOINT_COLOR = (255, 255, 255)
_DEFAULT_MIN_CONF = 0.3
_BONE_THICKNESS = 2
_JOINT_RADIUS = 3


def draw_pose(
    frame_rgb: np.ndarray,
    result: PoseResult,
    min_conf: float = _DEFAULT_MIN_CONF,
    bone_thickness: int = _BONE_THICKNESS,
    joint_radius: int = _JOINT_RADIUS,
) -> np.ndarray:
    """Return a copy of ``frame_rgb`` with every person's skeleton drawn."""
    canvas = np.ascontiguousarray(frame_rgb).copy()
    colors = bone_colors()

    for person in result.persons:
        xy = person.keypoints_xy
        conf = person.keypoints_conf
        for bone_idx, (a, b) in enumerate(COCO_SKELETON):
            if conf[a] < min_conf or conf[b] < min_conf:
                continue
            pa = (int(round(xy[a, 0])), int(round(xy[a, 1])))
            pb = (int(round(xy[b, 0])), int(round(xy[b, 1])))
            _draw_line(canvas, pa, pb, colors[bone_idx], bone_thickness)
        for k in range(xy.shape[0]):
            if conf[k] < min_conf:
                continue
            center = (int(round(xy[k, 0])), int(round(xy[k, 1])))
            _draw_point(canvas, center, _JOINT_COLOR, joint_radius)
    return canvas


def _draw_line(
    canvas: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if _HAS_CV2:
        cv2.line(canvas, p0, p1, color, thickness, lineType=cv2.LINE_AA)
        return
    _np_line(canvas, p0, p1, color, thickness)


def _draw_point(
    canvas: np.ndarray,
    center: tuple[int, int],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    if _HAS_CV2:
        cv2.circle(canvas, center, radius, color, -1, lineType=cv2.LINE_AA)
        return
    _np_disk(canvas, center, color, radius)


def _np_line(
    canvas: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Bresenham-ish line via dense sampling; fast enough for 17-bone skeletons."""
    x0, y0 = p0
    x1, y1 = p1
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1)
    ys = np.linspace(y0, y1, steps + 1)
    half = max(thickness // 2, 0)
    for x, y in zip(xs, ys):
        _np_disk(canvas, (int(round(x)), int(round(y))), color, half)


def _np_disk(
    canvas: np.ndarray,
    center: tuple[int, int],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    height, width = canvas.shape[0], canvas.shape[1]
    cx, cy = center
    x_lo, x_hi = max(cx - radius, 0), min(cx + radius + 1, width)
    y_lo, y_hi = max(cy - radius, 0), min(cy + radius + 1, height)
    if x_lo >= x_hi or y_lo >= y_hi:
        return
    canvas[y_lo:y_hi, x_lo:x_hi, 0] = color[0]
    canvas[y_lo:y_hi, x_lo:x_hi, 1] = color[1]
    canvas[y_lo:y_hi, x_lo:x_hi, 2] = color[2]
