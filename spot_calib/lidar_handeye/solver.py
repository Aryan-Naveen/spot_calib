"""AX = XB hand-eye solver for ``X = body_T_lidar``.

For stations i, j: ``A = body_i^-1 body_j`` (Spot odometry) and ``B = lidar_i^-1 lidar_j``
(scan registration, ``B = target_T_source`` with target = i). Then ``A X = X B``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from ..common import transforms as tf


@dataclass
class Pair:
    i: int
    j: int
    A: np.ndarray
    B: np.ndarray
    fitness: float = 1.0
    rmse: float = 0.0
    ok: bool = True

    def to_dict(self):
        return {"i": self.i, "j": self.j, "A": self.A.tolist(), "B": self.B.tolist(), "fitness": self.fitness,
                "rmse": self.rmse, "ok": self.ok}

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["j"], np.array(d["A"]), np.array(d["B"]), d["fitness"], d["rmse"], d["ok"])


def init_rotation(pairs: list[Pair], min_angle_deg: float = 2.0) -> np.ndarray | None:
    """Kabsch on rotation axes: rotvec(R_A) = R_X rotvec(R_B)."""
    a, b = [], []
    for p in pairs:
        ra, rb = tf.so3_log(p.A[:3, :3]), tf.so3_log(p.B[:3, :3])
        if np.degrees(np.linalg.norm(ra)) > min_angle_deg:
            a.append(ra)
            b.append(rb)
    if len(a) < 2:
        return None
    a, b = np.array(a), np.array(b)
    H = b.T @ a
    U, S, Vt = np.linalg.svd(H)
    if S[1] < 1e-3 * S[0]:  # all rotations about one axis
        return None
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def init_translation(pairs: list[Pair], R: np.ndarray, t_prior: np.ndarray, prior_w: float = 1e-3) -> np.ndarray:
    """(R_A - I) t = R t_B - t_A, with a tiny ridge towards the prior for unobservable directions."""
    M = [p.A[:3, :3] - np.eye(3) for p in pairs] + [prior_w * np.eye(3)]
    y = [R @ p.B[:3, 3] - p.A[:3, 3] for p in pairs] + [prior_w * t_prior]
    return np.linalg.lstsq(np.vstack(M), np.concatenate(y), rcond=None)[0]


def pair_residual(p: Pair, X: np.ndarray) -> tuple[float, float]:
    """(rotation residual [deg], translation residual [m]) of A X vs X B."""
    E = tf.inv(p.A @ X) @ (X @ p.B)
    return tf.rot_angle_deg(E[:3, :3]), float(np.linalg.norm(E[:3, 3]))


@dataclass
class HandEyeResult:
    X: np.ndarray
    sigma_t: float  # 1-sigma translation [m] (Gauss-Newton)
    sigma_r: float  # [deg]
    cov: np.ndarray
    res_rot_deg: np.ndarray
    res_trans_m: np.ndarray
    init: str


def solve(pairs: list[Pair], X_prior: np.ndarray, sigma_t: float = 0.01, sigma_r_deg: float = 0.2,
          prior_sigma_t: float = 0.3, prior_sigma_r_deg: float = 20.0, huber: float = 3.0) -> HandEyeResult:
    pairs = [p for p in pairs if p.ok]
    if not pairs:
        raise RuntimeError("no usable station pairs")
    R0 = init_rotation(pairs)
    how = "closed-form"
    if R0 is None:
        R0, how = X_prior[:3, :3], "prior rotation (rotations about a single axis only)"
    t0 = init_translation(pairs, R0, X_prior[:3, 3])
    X0 = tf.make_T(R0, t0)
    sr = np.radians(sigma_r_deg)
    w = np.array([np.sqrt(max(p.fitness, 0.05)) for p in pairs])

    def unpack(x):
        return X0 @ tf.se3_exp(x)

    def resid(x):
        X = unpack(x)
        out = []
        for p, wi in zip(pairs, w):
            e = tf.se3_log(tf.inv(p.A @ X) @ (X @ p.B))
            out.append(wi * np.concatenate([e[:3] / sr, e[3:] / sigma_t]))
        e = tf.se3_log(tf.inv(X_prior) @ X)  # weak prior keeps unobservable directions bounded
        out.append(np.concatenate([e[:3] / np.radians(prior_sigma_r_deg), e[3:] / prior_sigma_t]))
        return np.concatenate(out)

    r = least_squares(resid, np.zeros(6), loss="huber", f_scale=huber)
    X = unpack(r.x)
    J = r.jac
    cov = np.linalg.pinv(J.T @ J)
    rr = np.array([pair_residual(p, X) for p in pairs])
    return HandEyeResult(X, float(np.sqrt(np.trace(cov[3:, 3:]))), float(np.degrees(np.sqrt(np.trace(cov[:3, :3])))),
                         cov, rr[:, 0], rr[:, 1], how)


def observability(pairs: list[Pair]) -> dict:
    """Rotation-axis coverage and conditioning of the translation sub-problem."""
    pairs = [p for p in pairs if p.ok]
    cover = np.zeros(3)
    for p in pairs:
        rv = np.degrees(tf.so3_log(p.A[:3, :3]))
        cover = np.maximum(cover, np.abs(rv))
    if pairs:
        M = np.vstack([p.A[:3, :3] - np.eye(3) for p in pairs])
        _, s, Vt = np.linalg.svd(M)
    else:
        s, Vt = np.zeros(3), np.eye(3)
    weak = Vt[-1]
    hints = []
    names = ["roll (about body x)", "pitch (about body y)", "yaw (about body z)"]
    for k in range(3):
        if cover[k] < 10:
            hints.append(f"add more {names[k]}: max so far {cover[k]:.1f} deg")
    if s[-1] < 0.15:
        hints.append(f"translation weakly observed along body direction {np.round(weak, 2).tolist()}: "
                     "rotate about axes perpendicular to it (tilt the body in roll/pitch, not only yaw)")
    return {"axis_coverage_deg": cover.tolist(), "translation_singular_values": s.tolist(),
            "weak_direction": weak.tolist(), "hints": hints}


def leave_one_station_out(pairs: list[Pair], station_ids: list[int], X_prior: np.ndarray, **kw) -> dict:
    full = solve(pairs, X_prior, **kw).X
    devs = []
    for k in station_ids:
        sub = [p for p in pairs if p.ok and k not in (p.i, p.j)]
        if len(sub) < 3:
            continue
        try:
            devs.append(tf.se3_log(tf.inv(full) @ solve(sub, X_prior, **kw).X))
        except Exception:
            continue
    if len(devs) < 3:
        return {}
    v = np.array(devs)
    n = len(v)
    se = np.sqrt((n - 1) / n * np.sum((v - v.mean(axis=0)) ** 2, axis=0))
    return {"t_m": float(np.linalg.norm(se[3:])), "r_deg": float(np.degrees(np.linalg.norm(se[:3]))), "n": n}


def map_refine(X0: np.ndarray, pairs: list[Pair], clouds: dict[int, np.ndarray], outer: int = 5,
               max_dist: float = 0.3, n_src: int = 3000, seed: int = 0) -> np.ndarray:
    """Refine X by point-to-plane alignment of station clouds placed with odometry: for each pair,
    source points of station j are mapped into lidar frame i with X^-1 A X."""
    from .registration import estimate_normals

    rng = np.random.default_rng(seed)
    prep = {}
    for p in pairs:
        if not p.ok:
            continue
        tgt = clouds[p.i]
        if p.i not in prep:
            tree = cKDTree(tgt)
            n, planar = estimate_normals(tgt, tree)
            prep[p.i] = (tree, n, planar > 0.6)
    srcs = {p.j: clouds[p.j][rng.choice(len(clouds[p.j]), min(n_src, len(clouds[p.j])), replace=False)]
            for p in pairs if p.ok}
    X = X0.copy()
    for _ in range(outer):
        corr = []
        Xi = tf.inv(X)
        for p in pairs:
            if not p.ok:
                continue
            tree, nrm, good = prep[p.i]
            q = tf.transform_points(Xi @ p.A @ X, srcs[p.j])
            d, idx = tree.query(q, distance_upper_bound=max_dist)
            m = np.isfinite(d)
            m[m] &= good[idx[m]]
            corr.append((p, srcs[p.j][m], tree.data[idx[m]], nrm[idx[m]]))

        def resid(x):
            Xc = X @ tf.se3_exp(x)
            Xci = tf.inv(Xc)
            return np.concatenate([np.einsum("ij,ij->i", n, tf.transform_points(Xci @ p.A @ Xc, s) - q)
                                   for p, s, q, n in corr if len(s)])

        r = least_squares(resid, np.zeros(6), loss="huber", f_scale=0.03)
        X = X @ tf.se3_exp(r.x)
    return X
