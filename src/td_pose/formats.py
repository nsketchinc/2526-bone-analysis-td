"""Convert a :class:`PoseResult` into TouchDesigner-friendly outputs.

Two shapes are produced:

* :func:`result_to_channels` — a ``dict[str, list[float]]`` where each key is a
  CHOP channel name and each list holds one value per detected person. This maps
  cleanly onto a Script CHOP where ``numSamples == numPeople`` (the idiomatic TD
  instancing layout: sample index == person index).
* :func:`result_to_dict` / :func:`result_to_json` — a nested structure suitable
  for an OSC Out CHOP, a Text/Table DAT, or sending over the network.

Coordinate convention for outputs (controlled by ``normalize`` / ``flip_y``):
* ``normalize=True``  → x, y in ``[0, 1]`` (divided by width / height).
* ``flip_y=True``     → y measured from the bottom (TD texture origin), i.e.
  ``y_out = 1 - y_norm`` (or ``height - y_px``), so it lines up with TOP UVs.
"""

from __future__ import annotations

import json

from .keypoints import COCO_KEYPOINT_NAMES
from .pose_engine import Person, PoseResult

# Channel-name suffixes (kept as constants to avoid stringly-typed drift).
SUFFIX_TX = ":tx"
SUFFIX_TY = ":ty"
SUFFIX_CONF = ":conf"
CHAN_SCORE = "score"
CHAN_ID = "id"
CHAN_BBOX = ("bbox:x1", "bbox:y1", "bbox:x2", "bbox:y2")


def _transform_xy(
    x_px: float,
    y_px: float,
    width: int,
    height: int,
    normalize: bool,
    flip_y: bool,
) -> tuple[float, float]:
    """Apply normalization + vertical flip to one pixel coordinate."""
    if normalize:
        x_out = x_px / width if width else 0.0
        y_norm = y_px / height if height else 0.0
        y_out = (1.0 - y_norm) if flip_y else y_norm
    else:
        x_out = x_px
        y_out = (height - y_px) if flip_y else y_px
    return x_out, y_out


def channel_names() -> list[str]:
    """The full, ordered list of channels emitted by :func:`result_to_channels`."""
    names: list[str] = []
    for kp in COCO_KEYPOINT_NAMES:
        names.extend((kp + SUFFIX_TX, kp + SUFFIX_TY, kp + SUFFIX_CONF))
    names.append(CHAN_SCORE)
    names.append(CHAN_ID)
    names.extend(CHAN_BBOX)
    return names


def result_to_channels(
    result: PoseResult,
    normalize: bool = True,
    flip_y: bool = True,
) -> dict[str, list[float]]:
    """Build ``{channel_name: [value_per_person, ...]}``.

    The returned dict always contains every channel from :func:`channel_names`.
    With zero people detected, every channel maps to an empty list (the caller
    should then set ``numSamples = 0``).
    """
    channels: dict[str, list[float]] = {name: [] for name in channel_names()}
    width = result.image_width
    height = result.image_height

    for person in result.persons:
        _append_person(channels, person, width, height, normalize, flip_y)
    return channels


def _append_person(
    channels: dict[str, list[float]],
    person: Person,
    width: int,
    height: int,
    normalize: bool,
    flip_y: bool,
) -> None:
    for k, kp_name in enumerate(COCO_KEYPOINT_NAMES):
        x_px = float(person.keypoints_xy[k, 0])
        y_px = float(person.keypoints_xy[k, 1])
        x_out, y_out = _transform_xy(x_px, y_px, width, height, normalize, flip_y)
        channels[kp_name + SUFFIX_TX].append(x_out)
        channels[kp_name + SUFFIX_TY].append(y_out)
        channels[kp_name + SUFFIX_CONF].append(float(person.keypoints_conf[k]))

    channels[CHAN_SCORE].append(person.score)
    channels[CHAN_ID].append(float(person.track_id if person.track_id is not None else -1))

    x1, y1, x2, y2 = person.bbox_xyxy
    bx1, by1 = _transform_xy(x1, y1, width, height, normalize, flip_y)
    bx2, by2 = _transform_xy(x2, y2, width, height, normalize, flip_y)
    channels[CHAN_BBOX[0]].append(bx1)
    channels[CHAN_BBOX[1]].append(by1)
    channels[CHAN_BBOX[2]].append(bx2)
    channels[CHAN_BBOX[3]].append(by2)


def result_to_dict(
    result: PoseResult,
    normalize: bool = True,
    flip_y: bool = True,
) -> dict[str, object]:
    """Nested, JSON-serializable representation of the whole frame."""
    people: list[dict[str, object]] = []
    width = result.image_width
    height = result.image_height

    for person in result.persons:
        keypoints: list[dict[str, float]] = []
        for k, kp_name in enumerate(COCO_KEYPOINT_NAMES):
            x_out, y_out = _transform_xy(
                float(person.keypoints_xy[k, 0]),
                float(person.keypoints_xy[k, 1]),
                width,
                height,
                normalize,
                flip_y,
            )
            keypoints.append(
                {
                    "name": kp_name,
                    "x": x_out,
                    "y": y_out,
                    "conf": float(person.keypoints_conf[k]),
                }
            )
        people.append(
            {
                "id": person.track_id,
                "score": person.score,
                "bbox": list(person.bbox_xyxy),
                "keypoints": keypoints,
            }
        )

    return {
        "width": width,
        "height": height,
        "num_people": result.num_people,
        "normalized": normalize,
        "flip_y": flip_y,
        "people": people,
    }


def result_to_json(
    result: PoseResult,
    normalize: bool = True,
    flip_y: bool = True,
) -> str:
    """Compact JSON string of :func:`result_to_dict`."""
    return json.dumps(
        result_to_dict(result, normalize=normalize, flip_y=flip_y),
        separators=(",", ":"),
    )
