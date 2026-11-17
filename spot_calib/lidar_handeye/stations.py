"""Find stationary windows ("stations") in the odometry and build one dense cloud per station."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..common import transforms as tf
from .bag import Odom


@dataclass
class Station:
    id: int
    t0: float
    t1: float
    odom_T_body: np.ndarray
    spread_t: float = 0.0  # pose jitter within window [m]
    spread_r: float = 0.0  # [deg]
    points: np.ndarray | None = None  # (N,3) lidar frame, downsampled
    n_frames: int = 0
    excluded: bool = False
    meta: dict = field(default_factory=dict)


def detect_stations(odom: Odom, still_lin: float = 0.02, still_ang: float = 0.02, min_still: float = 1.5,
                    margin: float = 0.3, max_window: float = 3.0, min_sep_t: float = 0.03,
                    min_sep_r_deg: float = 1.0) -> list[Station]:
    still = (odom.lin < still_lin) & (odom.ang < still_ang)
    # Contiguous runs of `still`.
    runs, start = [], None
    for k, s in enumerate(still):
        if s and start is None:
            start = k
        if (not s or k == len(still) - 1) and start is not None:
            end = k if s else k - 1
            runs.append((start, end))
            start = None
    stations: list[Station] = []
    for a, b in runs:
        t0, t1 = odom.t[a] + margin, odom.t[b] - margin
        if t1 - t0 < min_still - 2 * margin:
            continue
        mid = 0.5 * (t0 + t1)
        t0, t1 = max(t0, mid - max_window / 2), min(t1, mid + max_window / 2)
        sel = (odom.t >= t0) & (odom.t <= t1)
        if sel.sum() < 2:
            continue
        Ts = odom.T[sel]
        Tm = tf.average_poses(Ts)
        spread_t = float(np.max(np.linalg.norm(Ts[:, :3, 3] - Tm[:3, 3], axis=1)))
        spread_r = float(max(tf.rot_angle_deg(Tm[:3, :3].T @ T[:3, :3]) for T in Ts))
        st = Station(len(stations), float(t0), float(t1), Tm, spread_t, spread_r)
        if stations:
            dt, dr = tf.pose_diff(stations[-1].odom_T_body, Tm)
            if dt < min_sep_t and dr < min_sep_r_deg:
                # Same pose as the previous station (robot paused twice): keep the longer one.
                if (t1 - t0) > (stations[-1].t1 - stations[-1].t0):
                    st.id = stations[-1].id
                    stations[-1] = st
                continue
        stations.append(st)
    return stations


def voxel_downsample(P: np.ndarray, voxel: float) -> np.ndarray:
    if len(P) == 0:
        return P
    keys = np.floor(P / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((counts.size, 3))
    np.add.at(out, inv, P)
    return out / counts[:, None]


def build_cloud(frames: list[np.ndarray], min_range: float = 0.8, max_range: float = 40.0, voxel: float = 0.05) -> np.ndarray:
    if not frames:
        return np.zeros((0, 3))
    P = np.concatenate(frames)
    r = np.linalg.norm(P, axis=1)
    P = P[(r > min_range) & (r < max_range)]
    return voxel_downsample(P, voxel)
