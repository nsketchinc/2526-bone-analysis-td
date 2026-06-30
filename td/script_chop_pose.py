"""Script CHOP DAT callbacks — emits per-person bone/keypoint channels.

Paste this into the DAT that drives the Script CHOP inside the POSE component.
All real work lives in the ``PoseExt`` extension on the parent component, so this
file stays a one-liner of glue.

Output channels (one sample per detected person, sample index == person index):
  <keypoint>:tx, <keypoint>:ty, <keypoint>:conf   (x17 COCO keypoints)
  score, id, bbox:x1, bbox:y1, bbox:x2, bbox:y2
"""


def onSetupParameters(scriptOp):  # noqa: ANN001, N802 - TD signature
    return


def onCook(scriptOp):  # noqa: ANN001, N802 - TD signature
    scriptOp.parent().ext.PoseExt.cook_chop(scriptOp)
    return
