"""Assemble and save ``POSE.tox`` from inside TouchDesigner.

TouchDesigner is a Windows/macOS application, so a binary ``.tox`` cannot be
generated on a headless Linux box. Instead, run this script *once* inside TD's
Textport to build the component programmatically and save it:

    >>> exec(open(r'C:/path/to/bone/td/build_tox.py').read())
    >>> comp = build(r'C:/path/to/bone')     # returns the POSE component
    # POSE.tox is written next to this repo (or pass out_dir=...).

What it builds (a Base COMP named ``POSE``):

    in1 (In TOP) ─┬─► pose_top (Script TOP)  ──► out_overlay (Out TOP)   skeleton overlay
                  └─ (read inside the extension)
    frametrigger (Constant CHOP = absTime.frame) ─► pose_chop (Script CHOP) ─► out_bones (Out CHOP)

The component carries a custom "Pose" parameter page and a promoted ``PoseExt``
extension. ``Srcpath`` is wired to this repo's ``src`` folder so ``import td_pose``
resolves inside TD's Python (install deps first: see README).
"""

from __future__ import annotations

import os


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _add_params(comp, src_path: str) -> None:
    """Create the custom 'Pose' parameter page with sensible defaults."""
    page = comp.appendCustomPage("Pose")

    model = page.appendStr("Modelpath", label="Model Path")[0]
    model.default = model.val = "yolo26n-pose.pt"

    conf = page.appendFloat("Confidence", label="Confidence")[0]
    conf.normMin, conf.normMax, conf.clampMin, conf.clampMax = 0.0, 1.0, True, True
    conf.default = conf.val = 0.25

    imgsz = page.appendInt("Imgsz", label="Inference Size")[0]
    imgsz.default = imgsz.val = 640

    device = page.appendStr("Device", label="Device (blank=auto)")[0]
    device.default = device.val = ""

    maxdet = page.appendInt("Maxdet", label="Max People")[0]
    maxdet.default = maxdet.val = 10

    tracking = page.appendToggle("Tracking", label="Track IDs")[0]
    tracking.default = tracking.val = False

    tracker = page.appendStr("Tracker", label="Tracker Cfg")[0]
    tracker.default = tracker.val = "bytetrack.yaml"

    normalize = page.appendToggle("Normalize", label="Normalize Coords")[0]
    normalize.default = normalize.val = True

    flipy = page.appendToggle("Flipy", label="Flip Y (TD UV)")[0]
    flipy.default = flipy.val = True

    minconf = page.appendFloat("Minconf", label="Overlay Min Conf")[0]
    minconf.normMin, minconf.normMax, minconf.clampMin, minconf.clampMax = 0.0, 1.0, True, True
    minconf.default = minconf.val = 0.3

    active = page.appendToggle("Active", label="Active")[0]
    active.default = active.val = True

    videotop = page.appendStr("Videotop", label="Source TOP")[0]
    videotop.default = videotop.val = "in1"

    srcpath = page.appendStr("Srcpath", label="td_pose src Path")[0]
    srcpath.default = srcpath.val = src_path


def build(project_dir: str, parent=None, out_dir: str | None = None):
    """Build the POSE component under ``parent`` and save ``POSE.tox``.

    Parameters
    ----------
    project_dir:
        Absolute path to this repository (the folder containing ``td/`` & ``src/``).
    parent:
        TD component to build inside. Defaults to ``op('/')`` (root).
    out_dir:
        Where to write ``POSE.tox``. Defaults to ``project_dir``.
    """
    parent = parent or op("/")  # noqa: F821 - TD global
    td_dir = os.path.join(project_dir, "td")
    src_path = os.path.join(project_dir, "src")
    out_dir = out_dir or project_dir

    # Fresh component.
    existing = parent.op("POSE")
    if existing is not None:
        existing.destroy()
    comp = parent.create(baseCOMP, "POSE")  # noqa: F821 - TD global
    comp.par.externaltox = ""

    # Nodes.
    in1 = comp.create(inTOP, "in1")  # noqa: F821
    pose_top = comp.create(scriptTOP, "pose_top")  # noqa: F821
    out_overlay = comp.create(outTOP, "out_overlay")  # noqa: F821
    frametrigger = comp.create(constantCHOP, "frametrigger")  # noqa: F821
    pose_chop = comp.create(scriptCHOP, "pose_chop")  # noqa: F821
    out_bones = comp.create(outCHOP, "out_bones")  # noqa: F821

    # DATs holding code.
    ext_dat = comp.create(textDAT, "PoseExt")  # noqa: F821
    ext_dat.text = _read(os.path.join(td_dir, "pose_ext.py"))
    top_cb = comp.create(textDAT, "top_callbacks")  # noqa: F821
    top_cb.text = _read(os.path.join(td_dir, "script_top_overlay.py"))
    chop_cb = comp.create(textDAT, "chop_callbacks")  # noqa: F821
    chop_cb.text = _read(os.path.join(td_dir, "script_chop_pose.py"))

    # Wiring.
    pose_top.inputConnectors[0].connect(in1)
    out_overlay.inputConnectors[0].connect(pose_top)
    pose_chop.inputConnectors[0].connect(frametrigger)
    out_bones.inputConnectors[0].connect(pose_chop)

    # frametrigger forces pose_chop to re-cook every frame.
    frametrigger.par.name0 = "frame"
    frametrigger.par.value0.expr = "absTime.frame"

    # Bind callbacks.
    pose_top.par.callbacks = top_cb.name
    pose_chop.par.callbacks = chop_cb.name

    # Parameters + extension.
    _add_params(comp, src_path)
    comp.par.extension1 = "op('PoseExt').module.PoseExt(me)"
    comp.par.promoteextension1 = True
    comp.initializeExtensions()

    # Tidy layout.
    for node, (x, y) in {
        in1: (-4, 2),
        pose_top: (-2, 2),
        out_overlay: (0, 2),
        frametrigger: (-4, -2),
        pose_chop: (-2, -2),
        out_bones: (0, -2),
        ext_dat: (-4, 0),
        top_cb: (-2, 0),
        chop_cb: (0, 0),
    }.items():
        node.nodeX, node.nodeY = x * 150, y * 120

    tox_path = os.path.join(out_dir, "POSE.tox")
    comp.save(tox_path)
    print(f"[build_tox] saved {tox_path}")
    print("[build_tox] wire your video into POSE/in1; bones on out_bones, overlay on out_overlay")
    return comp
