"""Standalone demo: run YOLO26 bone analysis on a webcam or video file.

This exercises the exact same ``td_pose`` engine the TouchDesigner component
uses, so you can validate inference and model download outside of TD.

    uv run python examples/run_webcam.py                 # webcam 0
    uv run python examples/run_webcam.py --source clip.mp4
    uv run python examples/run_webcam.py --source clip.mp4 --no-display --json out.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running straight from the repo without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_pose import (  # noqa: E402
    EngineConfig,
    PoseEngine,
    draw_pose,
    result_to_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26 bone analysis demo")
    parser.add_argument("--source", default="0", help="webcam index or video path")
    parser.add_argument("--model", default="yolo26n-pose.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="", help="'', 'cpu', '0', 'cuda:0', 'mps'")
    parser.add_argument("--track", action="store_true", help="enable persistent IDs")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--json", default="", help="write per-frame JSON lines here")
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    return cap


def main() -> None:
    args = parse_args()
    engine = PoseEngine(
        EngineConfig(
            model_path=args.model,
            confidence=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            enable_tracking=args.track,
        )
    )
    print(f"Loading model... (loaded: {engine.load()})")

    cap = open_capture(args.source)
    json_file = open(args.json, "w", encoding="utf-8") if args.json else None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = engine.infer(frame_rgb)

            if json_file is not None:
                json_file.write(result_to_json(result) + "\n")

            if not args.no_display:
                overlay_rgb = draw_pose(frame_rgb, result)
                overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    overlay_bgr,
                    f"people: {result.num_people}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("td_pose — bone analysis", overlay_bgr)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        cap.release()
        if json_file is not None:
            json_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
