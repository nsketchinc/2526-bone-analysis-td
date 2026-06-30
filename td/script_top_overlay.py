"""Script TOP DAT callbacks — draws the skeleton (bones) over the input video.

Paste this into the DAT that drives the Script TOP inside the POSE component.
The Script TOP outputs the source frame with every detected person's bones and
joints rendered on top, giving an immediate visual confirmation of the analysis.
"""


def onSetupParameters(scriptOp):  # noqa: ANN001, N802 - TD signature
    return


def onCook(scriptOp):  # noqa: ANN001, N802 - TD signature
    scriptOp.parent().ext.PoseExt.cook_top(scriptOp)
    return
