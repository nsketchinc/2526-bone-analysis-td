# td_pose — Real-time YOLO26 bone (pose) analysis for TouchDesigner

A TouchDesigner component (`POSE.tox`) that takes a **video TOP as input**, runs
**real-time human pose ("bone") analysis** in Python using the **Ultralytics
YOLO26 nano** model (`yolo26n-pose.pt`, 17 COCO keypoints), and outputs:

- **`out_bones` (CHOP)** — per-person keypoint coordinates, confidences, bbox and
  track id. One CHOP *sample per detected person*.
- **`out_overlay` (TOP)** — the input frame with the skeleton (bones + joints)
  drawn on top, for instant visual confirmation.

The inference engine (`src/td_pose`) is fully decoupled from TouchDesigner, so it
is unit-tested off-TD and reusable from plain Python (see `examples/run_webcam.py`).

---

## Why it's split this way

TouchDesigner is a Windows/macOS app and a `.tox` is a binary it authors itself,
so the repo ships **the engine + the TD glue + a builder script** rather than a
pre-baked binary:

| Path | Role |
|------|------|
| `src/td_pose/` | Pure-Python engine (numpy + lazy ultralytics). Testable anywhere. |
| `td/pose_ext.py` | `PoseExt` TD extension: persistent engine, per-frame cache, cook logic. |
| `td/script_chop_pose.py` | Script CHOP callback → bone channels. |
| `td/script_top_overlay.py` | Script TOP callback → skeleton overlay. |
| `td/build_tox.py` | Run inside TD to assemble & save `POSE.tox`. |
| `examples/run_webcam.py` | Off-TD demo (webcam/video) using the same engine. |
| `scripts/smoke_test.py` | One-shot real-model pipeline check. |

---

## Architecture / data flow

```
in1 (In TOP) ─┬─► pose_top (Script TOP) ──► out_overlay (Out TOP)   # skeleton overlay
              └─ frame read inside PoseExt
frametrigger (Constant CHOP = absTime.frame) ─► pose_chop (Script CHOP) ─► out_bones (Out CHOP)
```

- `TOP.numpyArray()` returns RGBA float32 `[0,1]`, origin **bottom-left**. The
  extension flips it vertically and converts RGB→BGR for Ultralytics.
- Inference runs **at most once per frame** (cached by `absTime.frame`); the CHOP
  and TOP share the cached result.
- `frametrigger` (a Constant CHOP whose value is `absTime.frame`) forces the
  Script CHOP to re-cook every frame even though it has no video input.

### `out_bones` channel layout (one sample per person)

```
<keypoint>:tx, <keypoint>:ty, <keypoint>:conf   # x17 COCO keypoints
score, id, bbox:x1, bbox:y1, bbox:x2, bbox:y2
```

`id` is `-1` when tracking is off. Coordinates are normalized `0..1` with Y in TD
UV space (bottom-left origin) by default — toggle with `Normalize` / `Flip Y`.

The 17 keypoints and 19 skeleton "bones" (COCO) are defined in
`src/td_pose/keypoints.py`.

---

## Setup

### 1. Install dependencies into TouchDesigner's Python

TouchDesigner ships its own Python (3.11 for TD 2023/2024). Install the deps into
**that** interpreter, or into a venv whose `site-packages` you add via the
component's `Srcpath` + `sys.path`. From a shell using TD's python:

```bash
# Use the Python that matches your TouchDesigner build.
python -m pip install ultralytics
```

- GPU: install a CUDA build of `torch`/`torchvision` matching your driver
  (https://pytorch.org). CPU works but is slower — lower `Inference Size` (e.g.
  `imgsz=320`) and `Max People` for real-time.
- `yolo26n-pose.pt` (~8 MB) auto-downloads on first inference. If your Ultralytics
  build predates YOLO26, the engine **falls back** to `yolo11n-pose.pt` →
  `yolov8n-pose.pt` automatically.

### 2. Build `POSE.tox`

In the TouchDesigner **Textport**:

```python
exec(open(r'C:/path/to/bone/td/build_tox.py').read())
comp = build(r'C:/path/to/bone')      # creates /POSE and saves POSE.tox
```

`build()` wires the component's `Srcpath` parameter to this repo's `src/` folder
so `import td_pose` resolves. Then:

1. Drag `POSE.tox` into your network (or use the `/POSE` it just created).
2. Wire your video (Movie File In TOP, Video Device In TOP, NDI, …) into
   `POSE/in1`.
3. Read bones from `POSE/out_bones`, overlay from `POSE/out_overlay`.

### Custom parameters (Pose page)

| Param | Default | Meaning |
|-------|---------|---------|
| Model Path | `yolo26n-pose.pt` | Any ultralytics pose checkpoint. |
| Confidence | `0.25` | Detection threshold. |
| Inference Size | `640` | `imgsz`; lower = faster. |
| Device | `` (auto) | `cpu`, `0`, `cuda:0`, `mps`. |
| Max People | `10` | `max_det`. |
| Track IDs | `off` | Persistent IDs via ByteTrack. |
| Normalize Coords | `on` | Output 0..1 vs pixels. |
| Flip Y (TD UV) | `on` | Bottom-left origin to match TOP UVs. |
| Overlay Min Conf | `0.3` | Hide low-confidence bones/joints. |
| Active | `on` | Master enable. |
| Source TOP | `in1` | Which TOP to read frames from. |
| td_pose src Path | auto | Added to `sys.path` for `import td_pose`. |

Editing any parameter rebuilds the engine on the next cook (no restart needed).

---

## Develop / test off TouchDesigner

```bash
uv sync --extra dev          # installs numpy, ultralytics, opencv, pytest
uv run pytest                # unit tests (model mocked — no torch needed to pass)
uv run python scripts/smoke_test.py        # real model on a synthetic frame
uv run python examples/run_webcam.py       # live webcam demo
uv run python examples/run_webcam.py --source clip.mp4 --no-display --json out.jsonl
```

> Unit tests mock Ultralytics, so they run fast and pass without torch installed.
> The smoke test and webcam demo download/load the real model.

---

## Notes / limitations

- Heavy CUDA `torch` wheels are large; for CPU-only dev use
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- The `.tox` depends on `td_pose` being importable at `Srcpath`; keep this repo's
  `src/` available on the machine running TD (or `pip install` this package into
  TD's Python and clear `Srcpath`).
- Pose models detect **people only** (COCO keypoints). For other skeletons, train
  a custom pose model and point `Model Path` at it.
