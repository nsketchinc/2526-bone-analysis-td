"""Configuration for the pose engine.

Kept as a plain, immutable dataclass so it is trivial to construct from
TouchDesigner custom parameters or from a CLI / test without importing torch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# Default model. YOLO26 nano pose; auto-downloads on first use.
DEFAULT_MODEL_PATH = "yolo26n-pose.pt"

# Ordered fallbacks if the running Ultralytics build predates the YOLO26 release.
FALLBACK_MODEL_PATHS: tuple[str, ...] = (
    "yolo11n-pose.pt",
    "yolov8n-pose.pt",
)

# Bounds (no magic numbers scattered in logic).
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMGSZ = 640
DEFAULT_MAX_DET = 10


@dataclass(frozen=True)
class EngineConfig:
    """Immutable inference settings."""

    model_path: str = DEFAULT_MODEL_PATH
    confidence: float = DEFAULT_CONFIDENCE
    imgsz: int = DEFAULT_IMGSZ
    # "" lets Ultralytics auto-select (CUDA if available else CPU).
    # Use "cpu", "0", "cuda:0", or "mps" to force a device.
    device: str = ""
    max_det: int = DEFAULT_MAX_DET
    # When tracking is enabled, persistent IDs are assigned across frames.
    enable_tracking: bool = False
    tracker: str = "bytetrack.yaml"

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError("model_path must not be empty")
        if not (MIN_CONFIDENCE <= self.confidence <= MAX_CONFIDENCE):
            raise ValueError(
                f"confidence must be in [{MIN_CONFIDENCE}, {MAX_CONFIDENCE}], "
                f"got {self.confidence}"
            )
        if self.imgsz <= 0:
            raise ValueError(f"imgsz must be positive, got {self.imgsz}")
        if self.max_det <= 0:
            raise ValueError(f"max_det must be positive, got {self.max_det}")

    def with_model(self, model_path: str) -> "EngineConfig":
        """Return a copy that uses a different model (used for fallback)."""
        return replace(self, model_path=model_path)
