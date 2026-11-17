"""ZED frame conventions (zed-ros2-wrapper URDF) used to convert the solved optical-frame pose
to ``<robot>_zed_camera_link``, which is the child frame in dcist's calibration.yaml.

zed_wrapper URDF chain: camera_link -> camera_center -> left_camera_frame -> left_camera_optical_frame

  camera_center      : xyz = (screw_offset_x, 0, height/2 - screw_offset_z), pitch = bottom_slope
  left_camera_frame  : xyz = (optical_offset_x, baseline/2, 0)
  left_optical_frame : rpy = (-pi/2, 0, -pi/2)

These defaults are believed correct for a zed2i but were NOT verified against an installed
zed_wrapper. Verify once on the robot with
    ros2 run tf2_ros tf2_echo <robot>_zed_camera_link <robot>_zed_left_camera_optical_frame
and, if it differs, pass the printed values via --zed-link-T-left-optical "x y z qx qy qz qw".
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..common import transforms as tf

ZED_URDF_PARAMS = {
    "zed2i": {"height": 0.03, "screw_offset_x": 0.0, "screw_offset_z": 0.0, "bottom_slope": 0.0,
              "optical_offset_x": -0.01, "baseline": 0.12},
    "zed2": {"height": 0.03, "screw_offset_x": 0.0, "screw_offset_z": 0.0, "bottom_slope": 0.05,
             "optical_offset_x": -0.01, "baseline": 0.12},
}


def link_T_left_optical(model: str = "zed2i", baseline: float | None = None) -> np.ndarray:
    p = dict(ZED_URDF_PARAMS[model])
    if baseline:
        p["baseline"] = baseline
    link_T_center = tf.make_T(Rotation.from_euler("y", p["bottom_slope"]).as_matrix(),
                              [p["screw_offset_x"], 0.0, p["height"] / 2 - p["screw_offset_z"]])
    center_T_left = tf.make_T(np.eye(3), [p["optical_offset_x"], p["baseline"] / 2, 0.0])
    return link_T_center @ center_T_left @ tf.FRAME_T_OPTICAL


def parse_override(s: str) -> np.ndarray:
    """'x y z qx qy qz qw' (tf2_echo order) -> 4x4."""
    v = [float(x) for x in s.replace(",", " ").split()]
    if len(v) != 7:
        raise ValueError("expected 7 numbers: x y z qx qy qz qw")
    return tf.make_T(Rotation.from_quat(v[3:]).as_matrix(), v[:3])
