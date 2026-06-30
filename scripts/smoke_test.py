"""End-to-end smoke test of the real model on a synthetic frame.

Downloads / loads ``yolo26n-pose.pt`` (or a fallback) and runs one inference on a
generated image, then prints the resolved model, people count, and a few output
channels. Exits 0 even if no people are detected (a blank frame legitimately has
none); exits non-zero only if the pipeline itself fails.

    uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_pose import (  # noqa: E402
    EngineConfig,
    PoseEngine,
    channel_names,
    draw_pose,
    result_to_channels,
)


def synthetic_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """A gradient frame with a crude figure — enough to exercise the pipeline."""
    rng = np.random.default_rng(0)
    frame = (rng.integers(40, 80, size=(height, width, 3))).astype(np.uint8)
    # A pale vertical bar in the middle (vaguely person-shaped silhouette).
    frame[80:400, 290:350] = 200
    frame[60:110, 300:340] = 210  # "head"
    return frame


def main() -> int:
    engine = PoseEngine(EngineConfig(device="cpu"))
    loaded = engine.load()
    print(f"resolved model: {loaded}")

    frame = synthetic_frame()
    result = engine.infer(frame)
    print(f"people detected: {result.num_people}")
    print(f"frame size: {result.image_width}x{result.image_height}")

    channels = result_to_channels(result)
    assert set(channels.keys()) == set(channel_names()), "channel set drifted"
    overlay = draw_pose(frame, result)
    assert overlay.shape == frame.shape, "overlay changed shape"

    print(f"channel count: {len(channels)} (expected {len(channel_names())})")
    print("smoke test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
