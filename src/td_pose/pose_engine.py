"""Thin, testable wrapper around an Ultralytics YOLO pose model.

The engine takes a single RGB frame (uint8, top-left origin, shape ``(H, W, 3)``)
and returns a :class:`PoseResult` describing every detected person's keypoints
and the bones (skeleton edges) connecting them.

Ultralytics / torch are imported lazily inside :meth:`PoseEngine.load` so this
module can be imported (and most of it unit-tested) without the heavy deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import FALLBACK_MODEL_PATHS, EngineConfig
from .keypoints import NUM_KEYPOINTS


@dataclass(frozen=True)
class Person:
    """One detected person.

    Attributes
    ----------
    keypoints_xy:
        ``(NUM_KEYPOINTS, 2)`` float32 pixel coordinates in the source image,
        top-left origin (x right, y down).
    keypoints_conf:
        ``(NUM_KEYPOINTS,)`` float32 per-keypoint confidence in ``[0, 1]``.
    bbox_xyxy:
        Person bounding box ``(x1, y1, x2, y2)`` in source pixels.
    score:
        Detection confidence for the whole person.
    track_id:
        Persistent track id when tracking is enabled, else ``None``.
    """

    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    track_id: int | None


@dataclass(frozen=True)
class PoseResult:
    """All people detected in one frame plus the frame dimensions."""

    persons: tuple[Person, ...]
    image_width: int
    image_height: int

    @property
    def num_people(self) -> int:
        return len(self.persons)


class PoseEngine:
    """Loads a YOLO pose model and runs inference on individual frames."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self._config = config or EngineConfig()
        self._model: Any | None = None
        self._loaded_model_path: str | None = None

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_model_path(self) -> str | None:
        return self._loaded_model_path

    def load(self) -> str:
        """Load the configured model, falling back to older pose checkpoints.

        Returns the path of the model that was actually loaded. Raised errors
        from the final fallback are propagated so failures are not silent.
        """
        from ultralytics import YOLO  # lazy: keeps torch out of import path

        candidates = [self._config.model_path, *FALLBACK_MODEL_PATHS]
        # Preserve order while removing duplicates (config may equal a fallback).
        seen: set[str] = set()
        ordered = [c for c in candidates if not (c in seen or seen.add(c))]

        last_error: Exception | None = None
        for path in ordered:
            try:
                self._model = YOLO(path)
                self._loaded_model_path = path
                return path
            except Exception as exc:  # noqa: BLE001 - try next checkpoint
                last_error = exc
        raise RuntimeError(
            f"Failed to load any pose model from {ordered}: {last_error}"
        )

    def infer(self, frame_rgb: np.ndarray) -> PoseResult:
        """Run pose estimation on one RGB frame.

        Parameters
        ----------
        frame_rgb:
            ``(H, W, 3)`` uint8 array, RGB channel order, top-left origin.
        """
        if self._model is None:
            self.load()

        self._validate_frame(frame_rgb)
        height, width = frame_rgb.shape[0], frame_rgb.shape[1]

        # Ultralytics expects BGR (cv2 convention) for numpy inputs.
        # Force a contiguous buffer: the reversed view has negative strides,
        # which some OpenCV / Ultralytics internals reject.
        frame_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])

        cfg = self._config
        if cfg.enable_tracking:
            raw = self._model.track(
                source=frame_bgr,
                imgsz=cfg.imgsz,
                conf=cfg.confidence,
                device=cfg.device or None,
                max_det=cfg.max_det,
                persist=True,
                tracker=cfg.tracker,
                verbose=False,
            )
        else:
            raw = self._model.predict(
                source=frame_bgr,
                imgsz=cfg.imgsz,
                conf=cfg.confidence,
                device=cfg.device or None,
                max_det=cfg.max_det,
                verbose=False,
            )

        persons = self._parse_result(raw[0]) if raw else ()
        return PoseResult(persons=persons, image_width=width, image_height=height)

    @staticmethod
    def _validate_frame(frame_rgb: np.ndarray) -> None:
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(
                f"frame_rgb must have shape (H, W, 3), got {frame_rgb.shape}"
            )

    @staticmethod
    def _parse_result(result: Any) -> tuple[Person, ...]:
        """Convert a single Ultralytics ``Results`` object into ``Person``s."""
        keypoints = getattr(result, "keypoints", None)
        boxes = getattr(result, "boxes", None)
        if keypoints is None or keypoints.xy is None:
            return ()

        xy = _to_numpy(keypoints.xy)  # (n, K, 2)
        if xy.size == 0:
            return ()
        conf = (
            _to_numpy(keypoints.conf)
            if getattr(keypoints, "conf", None) is not None
            else np.ones(xy.shape[:2], dtype=np.float32)
        )

        scores, ids = _extract_box_meta(boxes, num_people=xy.shape[0])
        bboxes = _extract_boxes(boxes, num_people=xy.shape[0])

        persons: list[Person] = []
        for i in range(xy.shape[0]):
            kp_xy = xy[i].astype(np.float32)
            kp_conf = conf[i].astype(np.float32)
            if kp_xy.shape[0] != NUM_KEYPOINTS:
                # Unexpected topology; skip rather than emit garbage channels.
                continue
            persons.append(
                Person(
                    keypoints_xy=kp_xy,
                    keypoints_conf=kp_conf,
                    bbox_xyxy=bboxes[i],
                    score=float(scores[i]),
                    track_id=ids[i],
                )
            )
        return tuple(persons)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor or array-like to a CPU numpy array."""
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        return np.asarray(numpy_fn())
    return np.asarray(value)


def _extract_box_meta(
    boxes: Any, num_people: int
) -> tuple[np.ndarray, list[int | None]]:
    """Return (scores, track_ids) aligned to ``num_people``."""
    scores = np.ones(num_people, dtype=np.float32)
    ids: list[int | None] = [None] * num_people
    if boxes is None:
        return scores, ids

    if getattr(boxes, "conf", None) is not None:
        conf = _to_numpy(boxes.conf).reshape(-1)
        for i in range(min(num_people, conf.shape[0])):
            scores[i] = float(conf[i])

    if getattr(boxes, "id", None) is not None:
        raw_ids = _to_numpy(boxes.id).reshape(-1)
        for i in range(min(num_people, raw_ids.shape[0])):
            ids[i] = int(raw_ids[i])
    return scores, ids


def _extract_boxes(
    boxes: Any, num_people: int
) -> list[tuple[float, float, float, float]]:
    """Return per-person (x1, y1, x2, y2); zeros when unavailable."""
    default: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return [default] * num_people

    xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
    result: list[tuple[float, float, float, float]] = []
    for i in range(num_people):
        if i < xyxy.shape[0]:
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            result.append((x1, y1, x2, y2))
        else:
            result.append(default)
    return result
