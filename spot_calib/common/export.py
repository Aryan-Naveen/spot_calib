"""calibration.yaml entry formatting, comparison against an existing file, and session dirs."""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path

import numpy as np
import yaml

from . import transforms as tf


def format_entry(frame_id: str, child_frame_id: str, T: np.ndarray) -> str:
    """Return one ``frames:`` list item exactly in the dcist calibration.yaml style."""
    q = tf.quat_wxyz(T[:3, :3])
    t = T[:3, 3]
    lines = [
        f"  - frame_id: {frame_id}",
        f"    child_frame_id: {child_frame_id}",
        f"    qw: {repr(float(q[0]))}",
        f"    qx: {repr(float(q[1]))}",
        f"    qy: {repr(float(q[2]))}",
        f"    qz: {repr(float(q[3]))}",
        f"    x: {repr(float(t[0]))}",
        f"    y: {repr(float(t[1]))}",
        f"    z: {repr(float(t[2]))}",
    ]
    return "\n".join(lines) + "\n"


def load_existing(path: str | None, frame_id: str, child_frame_id: str) -> np.ndarray | None:
    """Look up ``frame_id -> child_frame_id`` in a dcist calibration.yaml (None if absent)."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    for fr in data.get("frames", []):
        if fr.get("frame_id") == frame_id and fr.get("child_frame_id") == child_frame_id:
            R = tf.R_from_quat_wxyz(fr.get("qw", 1.0), fr.get("qx", 0.0), fr.get("qy", 0.0), fr.get("qz", 0.0))
            return tf.make_T(R, [fr.get("x", 0.0), fr.get("y", 0.0), fr.get("z", 0.0)])
    return None


def compare(T_new: np.ndarray, T_old: np.ndarray | None) -> dict | None:
    if T_old is None:
        return None
    dt, dr = tf.pose_diff(T_old, T_new)
    return {
        "old": tf.describe(T_old),
        "new": tf.describe(T_new),
        "delta_translation_m": dt,
        "delta_rotation_deg": dr,
        "delta_xyz_m": (T_new[:3, 3] - T_old[:3, 3]).tolist(),
    }


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def dump_json(path: str | Path, obj) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, cls=_Encoder)
    os.replace(tmp, path)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def new_session_dir(root: str | Path, tool: str) -> Path:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d = Path(root) / tool / stamp
    d.mkdir(parents=True, exist_ok=True)
    return d
