# Task: LOCAL-TDPOSE — TouchDesigner TOX for real-time YOLO26 bone (pose) analysis

## Meta
- linear_id: (none provided — local project; can be linked later)
- tier: L
- created: 2026-06-30
- status: done

## Brief
Build a TouchDesigner component (.tox) that takes a video TOP as input, runs
real-time human pose ("bone") analysis in Python using the Ultralytics
YOLO26 nano pose model (`yolo26n-pose.pt`, 17 COCO keypoints), and outputs the
bone/skeleton information so it is usable downstream inside TouchDesigner.

Outputs required:
- Per-person keypoint coordinates + confidence (the canonical pose/bone data).
- Skeleton (bone) edge definitions for drawing/derivation.
- A live skeleton overlay image (visual confirmation of "bone analysis").

Constraints:
- Runs on WSL2 Linux for development; TouchDesigner itself runs on Windows/macOS,
  so the .tox cannot be authored binary here. We deliver: (1) a decoupled,
  test-covered Python engine, (2) TouchDesigner glue scripts (Script CHOP /
  Script TOP / extension), and (3) a `build_tox.py` that assembles & saves the
  real `.tox` from inside TD's textport.

## Decision Log
- 2026-06-30: tier = L (Hard Trigger: new core dependency `ultralytics`/`torch`).
- 2026-06-30: Architecture = in-process inference inside TD Python (Script CHOP +
  Script TOP), engine decoupled into `src/td_pose` so it is unit-testable off-TD.
- 2026-06-30: Model `yolo26n-pose.pt` (auto-download), configurable + fallback to
  `yolo11n-pose.pt` / `yolov8n-pose.pt` if a TD Python lacks the YOLO26 release.
- 2026-06-30: CHOP output layout = samples-as-people (idiomatic TD instancing):
  `numSamples = numPeople`, channels `{kp}:tx/:ty/:conf` (+ score/id/bbox), with
  y flipped to TD bottom-left origin and coords normalized 0..1 by default.
- 2026-06-30: TD `TOP.numpyArray()` is RGBA float32 0..1, bottom-up → flip vert,
  convert RGB→BGR for Ultralytics, map keypoint y back for TD.

## Design
See `## Design` details captured in README.md (Architecture section) and module
docstrings. Data flow:

```
movie/video TOP ──► Script CHOP (PoseExt.cook_chop)
                       │  numpyArray() → flip → BGR → PoseEngine.infer()
                       │  PoseResult → result_to_channels() → CHOP channels
                       └─► bone data (per-person keypoints/conf/bbox/id)
                ──► Script TOP (PoseExt.cook_top)
                       └─► input image + skeleton bones drawn (overlay)
```

Modules:
- `td_pose/keypoints.py` — COCO 17 names + 19 skeleton bones + colors.
- `td_pose/config.py`   — `EngineConfig` (model, conf, imgsz, device, max_det, tracker).
- `td_pose/pose_engine.py` — `PoseEngine` wraps Ultralytics YOLO pose; `infer()`.
- `td_pose/formats.py`  — `PoseResult`→CHOP channel dict and `→json` (OSC/DAT).
- `td_pose/draw.py`     — numpy/cv2 skeleton overlay renderer.
- `td/pose_ext.py`      — TD extension: lazy engine, `cook_chop`, `cook_top`.
- `td/script_chop_pose.py`, `td/script_top_overlay.py` — thin TD callbacks.
- `td/build_tox.py`     — assemble + save `POSE.tox` from TD textport.

## Implementation Notes
Implemented (13 files):
- Engine: `src/td_pose/{keypoints,config,pose_engine,formats,draw,__init__}.py`.
- TD glue: `td/{pose_ext,script_chop_pose,script_top_overlay,build_tox}.py`.
- Off-TD: `examples/run_webcam.py`, `scripts/smoke_test.py`.
- Packaging/docs: `pyproject.toml`, `.gitignore`, `README.md`.
- Tests: `tests/{conftest,test_keypoints,test_config,test_formats,test_pose_engine,test_draw}.py`.

Key correctness handling:
- `TOP.numpyArray()` is RGBA float32 [0,1] bottom-up → flip vert + RGB→BGR (made
  contiguous; reversed view has negative strides that break OpenCV).
- Inference cached per `absTime.frame` so CHOP + TOP share one inference/frame.
- Script CHOP forced to re-cook each frame via `frametrigger` Constant CHOP
  (`value = absTime.frame`).
- Model fallback chain yolo26n → yolo11n → yolov8n pose if YOLO26 unavailable.

## Review
Self-review (tier-L would normally use 4 reviewers; ran Claude self-review +
real execution since no OpenCG/Gemini CLIs wired here):
- 32/32 unit tests pass (model mocked; also pass with real ultralytics installed).
- Real-model smoke test: `yolo26n-pose.pt` auto-downloaded (assets v8.4.0),
  inference ran on CPU.
- Real-image E2E (zidane.jpg 1280x720): 2 people, nose conf 0.997, score 0.903,
  overlay rendered correctly (`assets/zidane_overlay.jpg`).
- Security: no secrets, model name validated, no SQL/web surface, input frame
  shape validated. Errors in TD cook paths are caught so TD keeps cooking.
PASS.

## Deploy
Local git only (no remote configured). Work committed on feature branch
`feat/td-yolo-pose-tox`. To create the actual `POSE.tox`: open the project in
TouchDesigner and run `td/build_tox.py` (see README "Build POSE.tox").
