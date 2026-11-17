"""PnP + joint bundle refinement of Spot-camera <- ZED extrinsics.

Notation: ``Z`` is the ZED left optical frame, ``S`` a Spot camera sensor (optical) frame, ``W``
the checkerboard. Unknowns are ``S_T_Z`` for each Spot camera (``independent`` mode) or a single
``body_T_Z`` combined with Spot's factory ``body_T_S`` (``joint`` mode), plus one ``Z_T_W`` per
sample. Residuals are corner reprojection errors in pixels in every camera that saw the board.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from ..common import transforms as tf
from .camera import CameraModel

ZED = "zed"


@dataclass
class Sample:
    id: int
    corners: dict[str, np.ndarray]  # cam name -> (N,2) canonical corners (zed always present)
    body_T_cam: dict[str, np.ndarray] = field(default_factory=dict)  # spot cams, from snapshot
    parent_T_cam: dict[str, np.ndarray] = field(default_factory=dict)  # e.g. frontleft_T_frontleft_fisheye
    status: str = "pending"  # pending | approved | rejected

    def to_dict(self) -> dict:
        return {"id": self.id, "status": self.status,
                "corners": {k: v.tolist() for k, v in self.corners.items()},
                "body_T_cam": {k: v.tolist() for k, v in self.body_T_cam.items()},
                "parent_T_cam": {k: v.tolist() for k, v in self.parent_T_cam.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        arr = lambda m: {k: np.array(v, float) for k, v in m.items()}  # noqa: E731
        return cls(id=d["id"], status=d.get("status", "approved"), corners=arr(d["corners"]),
                   body_T_cam=arr(d.get("body_T_cam", {})), parent_T_cam=arr(d.get("parent_T_cam", {})))


def pnp(obj: np.ndarray, corners: np.ndarray, cam: CameraModel) -> tuple[np.ndarray, float]:
    """Board pose ``cam_T_board`` and RMS reprojection error [px]."""
    n = cam.undistort_norm(corners)
    I3 = np.eye(3)
    ok, rvec, tvec = cv2.solvePnP(obj, n, I3, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(obj, n, I3, None, flags=cv2.SOLVEPNP_ITERATIVE)
    rvec, tvec = cv2.solvePnPRefineLM(obj, n, I3, None, rvec, tvec)
    T = tf.rvec_tvec_to_T(rvec, tvec)
    err = cam.project(tf.transform_points(T, obj)) - corners
    return T, float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))


def per_sample_estimates(samples: list[Sample], cams: dict[str, CameraModel], obj: np.ndarray) -> dict:
    """For each sample: PnP poses, per-camera PnP RMS and ``S_T_Z`` per Spot camera."""
    out = {}
    for s in samples:
        poses, rms = {}, {}
        for c, uv in s.corners.items():
            poses[c], rms[c] = pnp(obj, uv, cams[c])
        est = {c: poses[c] @ tf.inv(poses[ZED]) for c in poses if c != ZED}
        out[s.id] = {"cam_T_board": poses, "pnp_rms": rms, "S_T_Z": est}
    return out


@dataclass
class Solution:
    mode: str
    S_T_Z: dict[str, np.ndarray]  # per spot cam (in joint mode derived from body_T_Z)
    body_T_Z: np.ndarray | None
    rms: dict[str, float]  # per camera
    sample_rms: dict[int, dict[str, float]]  # per sample, per camera
    sigma: dict[str, dict]  # per spot cam (or "body"): 1-sigma translation [m] / rotation [deg]
    Z_T_W: dict[int, np.ndarray]
    n_samples: int

    def summary(self) -> dict:
        return {"mode": self.mode, "rms": self.rms, "sigma": self.sigma, "n": self.n_samples,
                "S_T_Z": {k: tf.describe(v) for k, v in self.S_T_Z.items()},
                "body_T_Z": tf.describe(self.body_T_Z) if self.body_T_Z is not None else None}


def _static_body_T_cam(samples: list[Sample]) -> dict[str, np.ndarray]:
    acc: dict[str, list] = {}
    for s in samples:
        for c, T in s.body_T_cam.items():
            acc.setdefault(c, []).append(T)
    return {c: tf.average_poses(v) for c, v in acc.items()}


def solve(samples: list[Sample], cams: dict[str, CameraModel], obj: np.ndarray, mode: str = "independent",
          init: dict | None = None, huber_px: float = 1.0, max_nfev: int = 200) -> Solution | None:
    """Jointly refine extrinsics and board poses. ``init`` = per_sample_estimates output (optional)."""
    samples = [s for s in samples if ZED in s.corners]
    if not samples:
        return None
    init = init or per_sample_estimates(samples, cams, obj)
    spot_cams = sorted({c for s in samples for c in s.corners if c != ZED})
    if not spot_cams:
        return None
    body_T_S = _static_body_T_cam(samples)
    if mode == "joint":
        spot_cams = [c for c in spot_cams if c in body_T_S]
        if not spot_cams:
            return None
        cands = [body_T_S[c] @ init[s.id]["S_T_Z"][c] for s in samples for c in spot_cams if c in init[s.id]["S_T_Z"]]
        if not cands:
            return None
        X0 = {"body": tf.average_poses(cands)}
    else:
        X0 = {}
        for c in spot_cams:
            cands = [init[s.id]["S_T_Z"][c] for s in samples if c in init[s.id]["S_T_Z"]]
            if cands:
                X0[c] = tf.average_poses(cands)
        spot_cams = [c for c in spot_cams if c in X0]
    xkeys = list(X0)
    W0 = [init[s.id]["cam_T_board"][ZED] for s in samples]
    nx, ns = len(xkeys), len(samples)

    # Residual layout: for each sample, for each cam that saw it: 2N residuals.
    blocks = []  # (sample_idx, cam)
    for i, s in enumerate(samples):
        for c in s.corners:
            if c == ZED or c in spot_cams:
                blocks.append((i, c))
    N = len(obj)

    def S_T_Z_of(c, Xs):
        if mode == "joint":
            return tf.inv(body_T_S[c]) @ Xs["body"]
        return Xs[c]

    def unpack(p):
        Xs = {k: X0[k] @ tf.se3_exp(p[6 * j:6 * j + 6]) for j, k in enumerate(xkeys)}
        Ws = [W0[i] @ tf.se3_exp(p[6 * (nx + i):6 * (nx + i) + 6]) for i in range(ns)]
        return Xs, Ws

    def residuals(p):
        Xs, Ws = unpack(p)
        out = []
        for i, c in blocks:
            T = Ws[i] if c == ZED else S_T_Z_of(c, Xs) @ Ws[i]
            out.append((cams[c].project(tf.transform_points(T, obj)) - samples[i].corners[c]).ravel())
        return np.concatenate(out)

    m = 2 * N * len(blocks)
    sp = lil_matrix((m, 6 * (nx + ns)), dtype=int)
    for b, (i, c) in enumerate(blocks):
        r = slice(2 * N * b, 2 * N * (b + 1))
        sp[r, 6 * (nx + i):6 * (nx + i) + 6] = 1
        if c != ZED:
            j = 0 if mode == "joint" else xkeys.index(c)
            sp[r, 6 * j:6 * j + 6] = 1

    res = least_squares(residuals, np.zeros(6 * (nx + ns)), jac_sparsity=sp, loss="huber", f_scale=huber_px,
                        x_scale="jac", max_nfev=max_nfev, method="trf")
    Xs, Ws = unpack(res.x)
    r = res.fun.reshape(-1, N, 2)

    rms: dict[str, list] = {}
    sample_rms: dict[int, dict[str, float]] = {}
    for b, (i, c) in enumerate(blocks):
        e2 = np.sum(r[b] ** 2, axis=1)
        rms.setdefault(c, []).append(e2)
        sample_rms.setdefault(samples[i].id, {})[c] = float(np.sqrt(e2.mean()))
    rms_out = {c: float(np.sqrt(np.concatenate(v).mean())) for c, v in rms.items()}

    # Gauss-Newton covariance of the extrinsic blocks (marginalised over board poses).
    sigma = {}
    try:
        J = res.jac.toarray() if hasattr(res.jac, "toarray") else res.jac
        dof = max(1, m - J.shape[1])
        s2 = float(np.sum(res.fun ** 2) / dof)
        cov = np.linalg.pinv(J.T @ J) * s2
        for j, k in enumerate(xkeys):
            C = cov[6 * j:6 * j + 6, 6 * j:6 * j + 6]
            sigma[k] = {"t_m": float(np.sqrt(np.trace(C[3:, 3:]))),
                        "r_deg": float(np.degrees(np.sqrt(np.trace(C[:3, :3]))))}
    except Exception:
        pass

    S_T_Z = {c: S_T_Z_of(c, Xs) for c in spot_cams}
    return Solution(mode=mode, S_T_Z=S_T_Z, body_T_Z=Xs.get("body"), rms=rms_out, sample_rms=sample_rms,
                    sigma=sigma, Z_T_W={s.id: W for s, W in zip(samples, Ws)}, n_samples=ns)


def leave_one_out(samples: list[Sample], cams, obj, mode: str, init: dict) -> dict[str, dict]:
    """Spread (std) of the extrinsic when each sample is left out: {cam: {t_m, r_deg}}."""
    if len(samples) < 4:
        return {}
    full = solve(samples, cams, obj, mode, init)
    if full is None:
        return {}
    ests: dict[str, list] = {}
    for k in range(len(samples)):
        sub = samples[:k] + samples[k + 1:]
        sol = solve(sub, cams, obj, mode, init, max_nfev=60)
        if sol is None:
            continue
        keys = {"body": sol.body_T_Z} if mode == "joint" else sol.S_T_Z
        ref = {"body": full.body_T_Z} if mode == "joint" else full.S_T_Z
        for c, T in keys.items():
            if c in ref and T is not None:
                D = tf.inv(ref[c]) @ T
                ests.setdefault(c, []).append(tf.se3_log(D))
    out = {}
    n = len(samples)
    for c, v in ests.items():
        v = np.array(v)
        # Jackknife standard error: sqrt((n-1)/n * sum (x_i - mean)^2)
        dev = v - v.mean(axis=0)
        se = np.sqrt((n - 1) / n * np.sum(dev ** 2, axis=0))
        out[c] = {"t_m": float(np.linalg.norm(se[3:])), "r_deg": float(np.degrees(np.linalg.norm(se[:3])))}
    return out


def cross_camera_agreement(sol: Solution, samples: list[Sample]) -> tuple[float, float] | None:
    """body_T_Z implied by frontleft vs frontright (independent mode): (dt [m], dr [deg])."""
    body_T_S = _static_body_T_cam(samples)
    ests = [body_T_S[c] @ T for c, T in sol.S_T_Z.items() if c in body_T_S]
    if len(ests) < 2:
        return None
    return tf.pose_diff(ests[0], ests[1])


def coverage(samples: list[Sample], cams: dict[str, CameraModel], gx: int = 6, gy: int = 4) -> dict[str, list]:
    out = {}
    for c, cam in cams.items():
        g = np.zeros((gy, gx), int)
        for s in samples:
            if c in s.corners:
                uv = s.corners[c]
                ix = np.clip((uv[:, 0] / cam.width * gx).astype(int), 0, gx - 1)
                iy = np.clip((uv[:, 1] / cam.height * gy).astype(int), 0, gy - 1)
                cells = set(zip(iy.tolist(), ix.tolist()))
                for (a, b) in cells:
                    g[a, b] += 1
        out[c] = g.tolist()
    return out
