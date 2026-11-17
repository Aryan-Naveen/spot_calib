"""Scan-to-scan registration: small_gicp (GICP) if installed, else a scipy point-to-plane ICP."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..common import transforms as tf
from .stations import voxel_downsample

log = logging.getLogger(__name__)

try:
    import small_gicp  # type: ignore

    HAVE_SMALL_GICP = True
except Exception:  # pragma: no cover
    HAVE_SMALL_GICP = False


@dataclass
class RegResult:
    T: np.ndarray  # target_T_source
    fitness: float  # fraction of source points within inlier_dist of target
    rmse: float  # rmse of inlier distances [m]
    backend: str


def estimate_normals(P: np.ndarray, tree: cKDTree, k: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Per-point normals and planarity (1 - l0/l1) from k-NN PCA."""
    k = min(k, len(P))
    _, idx = tree.query(P, k=k)
    Q = P[idx] - P[idx].mean(axis=1, keepdims=True)
    C = np.einsum("nki,nkj->nij", Q, Q) / k
    w, v = np.linalg.eigh(C)
    n = v[:, :, 0]
    planarity = 1.0 - w[:, 0] / np.maximum(w[:, 1], 1e-12)
    return n, planarity


def _icp_point_to_plane(target: np.ndarray, source: np.ndarray, init: np.ndarray, stages) -> np.ndarray:
    T = init.copy()
    for voxel, max_dist, iters in stages:
        tgt = voxel_downsample(target, voxel)
        src = voxel_downsample(source, voxel)
        tree = cKDTree(tgt)
        nrm, planar = estimate_normals(tgt, tree)
        good = planar > 0.6
        for _ in range(iters):
            p = tf.transform_points(T, src)
            d, j = tree.query(p, distance_upper_bound=max_dist)
            m = np.isfinite(d)
            m[m] &= good[j[m]]
            if m.sum() < 50:
                break
            pm, q, n = p[m], tgt[j[m]], nrm[j[m]]
            r = np.einsum("ij,ij->i", n, pm - q)
            J = np.hstack([np.cross(pm, n), n])
            s = max(1e-3, 1.4826 * np.median(np.abs(r)))
            w = np.minimum(1.0, (1.5 * s) / np.maximum(np.abs(r), 1e-12))  # Huber
            H = J.T @ (J * w[:, None])
            g = J.T @ (w * r)
            dx = -np.linalg.solve(H + 1e-9 * np.eye(6), g)
            T = tf.se3_exp(dx) @ T
            if np.linalg.norm(dx[:3]) < 1e-5 and np.linalg.norm(dx[3:]) < 1e-5:
                break
    return T


def score(target_tree: cKDTree, source: np.ndarray, T: np.ndarray, inlier_dist: float = 0.1) -> tuple[float, float]:
    d, _ = target_tree.query(tf.transform_points(T, source), distance_upper_bound=inlier_dist)
    m = np.isfinite(d)
    return float(m.mean()) if len(d) else 0.0, float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else float("nan")


def register(target: np.ndarray, source: np.ndarray, init: np.ndarray, backend: str = "auto",
             threads: int = 4) -> RegResult:
    if backend == "auto":
        backend = "small_gicp" if HAVE_SMALL_GICP else "scipy"
    if backend == "small_gicp":
        if not HAVE_SMALL_GICP:
            raise RuntimeError("small_gicp not installed (pip install small_gicp)")
        T = init
        for res, maxd in [(0.25, 1.0), (0.1, 0.3)]:
            r = small_gicp.align(target, source, init_T_target_source=T, registration_type="GICP",
                                 downsampling_resolution=res, max_correspondence_distance=maxd, num_threads=threads)
            T = np.asarray(r.T_target_source)
    else:
        T = _icp_point_to_plane(target, source, init, [(0.3, 1.0, 30), (0.1, 0.3, 30), (0.05, 0.15, 20)])
    fit, rmse = score(cKDTree(target), voxel_downsample(source, 0.1), T)
    return RegResult(T, fit, rmse, backend)
