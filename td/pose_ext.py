"""PoseExt — TouchDesigner extension that drives real-time YOLO26 bone analysis.

Attach this as an extension on the component that becomes ``POSE.tox``. It owns a
single persistent :class:`td_pose.PoseEngine`, reads its settings from the
component's custom parameters, and exposes two cook entry points:

* :meth:`cook_chop` — called by the Script CHOP; writes per-person keypoint /
  bone channels.
* :meth:`cook_top`  — called by the Script TOP; draws the skeleton overlay.

Inference for a given video frame happens at most once: the result is cached by
TouchDesigner's global frame number and reused by whichever node cooks next.

This module references TouchDesigner globals (``op``, ``absTime``, ``project``,
``debug``) only *inside* methods, so it imports cleanly under plain CPython too.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np


def _ensure_td_pose_on_path(src_path: str) -> None:
    """Make ``import td_pose`` work by adding the package's ``src`` dir."""
    if src_path and src_path not in sys.path:
        sys.path.insert(0, src_path)


class PoseExt:
    """Owner-component extension. ``ownerComp`` is passed by TD on creation."""

    def __init__(self, ownerComp: Any) -> None:
        self.ownerComp = ownerComp
        self._engine: Any = None
        self._engine_signature: tuple[Any, ...] | None = None
        self._cache_frame: int = -1
        self._cache_key: str = ""
        self._cache_result: Any = None

    # ----- configuration -------------------------------------------------
    def _par(self, name: str, default: Any) -> Any:
        par = getattr(self.ownerComp.par, name, None)
        if par is None:
            return default
        return par.eval()

    def _build_config(self) -> Any:
        _ensure_td_pose_on_path(self._par("Srcpath", ""))
        from td_pose import EngineConfig

        return EngineConfig(
            model_path=self._par("Modelpath", "yolo26n-pose.pt"),
            confidence=float(self._par("Confidence", 0.25)),
            imgsz=int(self._par("Imgsz", 640)),
            device=str(self._par("Device", "")),
            max_det=int(self._par("Maxdet", 10)),
            enable_tracking=bool(self._par("Tracking", False)),
            tracker=str(self._par("Tracker", "bytetrack.yaml")),
        )

    def _config_signature(self, config: Any) -> tuple[Any, ...]:
        return (
            config.model_path,
            config.confidence,
            config.imgsz,
            config.device,
            config.max_det,
            config.enable_tracking,
            config.tracker,
        )

    def engine(self) -> Any:
        """Return the cached engine, rebuilding it if parameters changed."""
        config = self._build_config()
        signature = self._config_signature(config)
        if self._engine is None or signature != self._engine_signature:
            from td_pose import PoseEngine

            self._engine = PoseEngine(config)
            loaded = self._engine.load()
            self._engine_signature = signature
            self._log(f"PoseEngine loaded model: {loaded}")
        return self._engine

    def reload(self) -> None:
        """Force the engine to be rebuilt on next cook (e.g. after a param edit)."""
        self._engine = None
        self._engine_signature = None
        self._cache_frame = -1
        self._cache_result = None

    # ----- frame access --------------------------------------------------
    def _source_top(self) -> Any:
        path = self._par("Videotop", "in1")
        top = self.ownerComp.op(path)
        if top is None:
            raise RuntimeError(f"Videotop '{path}' not found under {self.ownerComp}")
        return top

    @staticmethod
    def _top_to_rgb_uint8(top: Any) -> np.ndarray:
        """TD TOP -> (H, W, 3) uint8 RGB, top-left origin.

        ``numpyArray`` yields RGBA float32 in [0, 1] with origin bottom-left, so
        we drop alpha, flip vertically, and scale to 8-bit.
        """
        arr = top.numpyArray(delayed=True)  # (H, W, 4) float32, bottom-up
        rgb_bottom_up = arr[:, :, :3]
        rgb_top_left = np.flipud(rgb_bottom_up)
        return (np.clip(rgb_top_left, 0.0, 1.0) * 255.0).astype(np.uint8)

    # ----- inference (cached per frame) ----------------------------------
    def result_for(self, top: Any, frame_rgb: np.ndarray) -> Any:
        key = str(top.path) if hasattr(top, "path") else str(id(top))
        frame = self._current_frame()
        if (
            self._cache_result is not None
            and frame == self._cache_frame
            and key == self._cache_key
        ):
            return self._cache_result

        result = self.engine().infer(frame_rgb)
        self._cache_frame = frame
        self._cache_key = key
        self._cache_result = result
        return result

    @staticmethod
    def _current_frame() -> int:
        try:
            return int(absTime.frame)  # noqa: F821 - TD global
        except Exception:  # noqa: BLE001
            return -1

    # ----- Script CHOP entry point ---------------------------------------
    def cook_chop(self, scriptOp: Any) -> None:
        scriptOp.clear()
        if not bool(self._par("Active", True)):
            scriptOp.numSamples = 0
            return

        from td_pose import channel_names, result_to_channels

        # Pre-create channels so output topology is stable even with 0 people.
        for name in channel_names():
            scriptOp.appendChan(name)

        try:
            top = self._source_top()
            frame_rgb = self._top_to_rgb_uint8(top)
            result = self.result_for(top, frame_rgb)
        except Exception as exc:  # noqa: BLE001 - keep TD cooking
            self._log(f"cook_chop error: {exc}")
            scriptOp.numSamples = 0
            return

        channels = result_to_channels(
            result,
            normalize=bool(self._par("Normalize", True)),
            flip_y=bool(self._par("Flipy", True)),
        )
        num_people = result.num_people
        scriptOp.numSamples = num_people
        if num_people == 0:
            return
        for name, values in channels.items():
            chan = scriptOp[name]
            for i, value in enumerate(values):
                chan[i] = value

    # ----- Script TOP entry point ----------------------------------------
    def cook_top(self, scriptOp: Any) -> None:
        from td_pose import draw_pose

        try:
            top = self._source_top()
            frame_rgb = self._top_to_rgb_uint8(top)
        except Exception as exc:  # noqa: BLE001
            self._log(f"cook_top error: {exc}")
            return

        if not bool(self._par("Active", True)):
            self._write_rgb(scriptOp, frame_rgb)
            return

        try:
            result = self.result_for(top, frame_rgb)
            drawn = draw_pose(
                frame_rgb, result, min_conf=float(self._par("Minconf", 0.3))
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"cook_top inference error: {exc}")
            drawn = frame_rgb
        self._write_rgb(scriptOp, drawn)

    @staticmethod
    def _write_rgb(scriptOp: Any, rgb_uint8: np.ndarray) -> None:
        """Write (H, W, 3) uint8 top-left RGB back to the Script TOP (RGBA float)."""
        rgb = rgb_uint8.astype(np.float32) / 255.0
        rgb_bottom_up = np.flipud(rgb)
        alpha = np.ones(rgb_bottom_up.shape[:2] + (1,), dtype=np.float32)
        rgba = np.concatenate([rgb_bottom_up, alpha], axis=2)
        scriptOp.copyNumpyArray(np.ascontiguousarray(rgba))

    # ----- misc ----------------------------------------------------------
    def _log(self, message: str) -> None:
        try:
            debug(message)  # noqa: F821 - TD global
        except Exception:  # noqa: BLE001
            print(message)
