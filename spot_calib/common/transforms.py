"""SE(3) helpers. All transforms are 4x4 numpy arrays, ``a_T_b`` maps points in b to a."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def so3_exp(w) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(w, dtype=float)).as_matrix()


def so3_log(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def se3_exp(xi) -> np.ndarray:
    """xi = [rotvec(3), translation(3)] (decoupled, not the true SE3 exponential)."""
    xi = np.asarray(xi, dtype=float)
    return make_T(so3_exp(xi[:3]), xi[3:6])


def se3_log(T: np.ndarray) -> np.ndarray:
    return np.concatenate([so3_log(T[:3, :3]), T[:3, 3]])


def rvec_tvec_to_T(rvec, tvec) -> np.ndarray:
    return make_T(so3_exp(np.asarray(rvec).reshape(3)), np.asarray(tvec).reshape(3))


def quat_wxyz(R: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def R_from_quat_wxyz(qw, qx, qy, qz) -> np.ndarray:
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(so3_log(R))))


def pose_diff(Ta: np.ndarray, Tb: np.ndarray) -> tuple[float, float]:
    """(translation difference [m], rotation difference [deg]) between two poses."""
    D = inv(Ta) @ Tb
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), rot_angle_deg(D[:3, :3])


def average_poses(Ts) -> np.ndarray:
    Ts = list(Ts)
    R = Rotation.from_matrix(np.stack([T[:3, :3] for T in Ts])).mean().as_matrix()
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    return make_T(R, t)


def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def to_list(T: np.ndarray) -> list:
    return np.asarray(T).tolist()


def describe(T: np.ndarray) -> dict:
    """Human-readable pose: xyz, quaternion (wxyz) and roll/pitch/yaw in degrees."""
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    return {
        "xyz": [float(v) for v in T[:3, 3]],
        "quat_wxyz": [float(v) for v in quat_wxyz(T[:3, :3])],
        "rpy_deg": [float(v) for v in rpy],
    }


# Rotation from a ROS "camera_frame" (x fwd, y left, z up) to its optical frame (x right, y down,
# z fwd): this is the rpy="-pi/2 0 -pi/2" joint used by every ROS camera description.
FRAME_T_OPTICAL = make_T(Rotation.from_euler("xyz", [-np.pi / 2, 0, -np.pi / 2]).as_matrix(), [0, 0, 0])
