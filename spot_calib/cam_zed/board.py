"""Checkerboard detection with a canonical (camera-independent) corner ordering.

OpenCV may return the corner grid in any of its symmetric orderings (180 deg rotation, and for
some detectors mirrored), and it can pick different ones in different cameras viewing the same
board. That would silently corrupt the extrinsic, so every detection is canonicalised:

1. Handedness: choose the ordering whose board x (along a row) and y (down the columns) axes form a
   right-handed frame with z pointing away from the camera. A camera can only see the printed face
   of the board, so this removes the mirrored orderings.
2. Colour: of the two remaining orderings (related by 180 deg), keep the one where the outer square
   diagonally beyond corner 0 is *dark*. For a board whose square counts make the 180 deg rotation
   map black to white (e.g. 11x8 squares = 10x7 inner corners) this is unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Board:
    cols: int = 10  # inner corners per row
    rows: int = 7  # inner corners per column
    square: float = 0.036  # metres

    @property
    def size(self) -> tuple[int, int]:
        return (self.cols, self.rows)

    @property
    def n(self) -> int:
        return self.cols * self.rows

    def object_points(self) -> np.ndarray:
        """(N,3) board-frame corner coordinates, row-major (matching the canonical ordering)."""
        ys, xs = np.mgrid[0:self.rows, 0:self.cols]
        return np.stack([xs.ravel() * self.square, ys.ravel() * self.square, np.zeros(self.n)], axis=1).astype(float)

    @property
    def disambiguable(self) -> bool:
        # Squares are (cols+1) x (rows+1). Square (0,0) vs square (cols, rows) colour differ iff
        # cols+rows is odd.
        return (self.cols + self.rows) % 2 == 1


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _sample(gray: np.ndarray, p: np.ndarray, r: int) -> float | None:
    x, y = int(round(p[0])), int(round(p[1]))
    h, w = gray.shape
    if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
        return None
    return float(gray[y - r:y + r + 1, x - r:x + r + 1].mean())


def _outer_intensity(gray: np.ndarray, grid: np.ndarray) -> tuple[float | None, float | None]:
    """Mean intensity of the outer square beyond grid[0,0] and beyond grid[-1,-1]."""
    out = []
    for g in (grid, grid[::-1, ::-1]):
        c = g[0, 0]
        d = 0.5 * ((c - g[0, 1]) + (c - g[1, 0]))  # half a square diagonally outward
        r = max(1, int(0.25 * min(np.linalg.norm(g[0, 1] - c), np.linalg.norm(g[1, 0] - c))))
        out.append(_sample(gray, c + d, r))
    return out[0], out[1]


def _handedness(grid: np.ndarray) -> float:
    dx = (grid[:, -1] - grid[:, 0]).mean(axis=0)
    dy = (grid[-1, :] - grid[0, :]).mean(axis=0)
    return float(dx[0] * dy[1] - dx[1] * dy[0])


def canonicalize(gray: np.ndarray, corners: np.ndarray, board: Board) -> tuple[np.ndarray | None, str]:
    """Return canonically ordered (N,2) corners, or (None, reason)."""
    grid = corners.reshape(board.rows, board.cols, 2)
    cands = [grid, grid[::-1, ::-1], grid[::-1, :], grid[:, ::-1]]
    cands = [g for g in cands if _handedness(g) > 0]
    if len(cands) != 2:
        return None, "degenerate grid"
    if not board.disambiguable:
        return cands[0].reshape(-1, 2), "ambiguous board (180 deg symmetric)"
    a, b = _outer_intensity(gray, cands[0])
    if a is None or b is None:
        return None, "board corner squares out of view"
    if abs(a - b) < 25:
        return None, f"low contrast at board corners ({a:.0f} vs {b:.0f})"
    g = cands[0] if a < b else cands[1]
    return np.ascontiguousarray(g.reshape(-1, 2)), ""


def detect(img: np.ndarray, board: Board, fast: bool = False) -> tuple[np.ndarray | None, str]:
    """Detect and canonicalise corners. ``fast`` is for live preview (lower accuracy)."""
    gray = _to_gray(img)
    corners = None
    if fast:
        # The classic detector misses boards whose squares run to the board edge (no white
        # margin); the sector-based detector handles them and is fast on preview-sized images.
        ok, c = cv2.findChessboardCornersSB(gray, board.size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE |
                                            cv2.CALIB_CB_EXHAUSTIVE)
        if ok:
            corners = c.reshape(-1, 2).astype(np.float64)
    else:
        ok, c = cv2.findChessboardCornersSB(gray, board.size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE |
                                            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            corners = c.reshape(-1, 2).astype(np.float32)
        else:
            ok, c = cv2.findChessboardCorners(gray, board.size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH |
                                              cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ok:
                c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
                corners = c.reshape(-1, 2)
    if corners is None:
        return None, "board not found"
    return canonicalize(gray, corners.astype(np.float64), board)


def draw(img: np.ndarray, corners: np.ndarray | None, board: Board, msg: str = "") -> np.ndarray:
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif vis.shape[2] == 4:
        vis = cv2.cvtColor(vis, cv2.COLOR_BGRA2BGR)
    th = max(1, vis.shape[1] // 640)
    if corners is not None:
        cv2.drawChessboardCorners(vis, board.size, corners.reshape(-1, 1, 2).astype(np.float32), True)
        # Mark the canonical origin and x axis so the operator can see orderings agree.
        o, x = corners[0], corners[board.cols - 1]
        cv2.circle(vis, tuple(int(v) for v in o), 8 * th, (0, 0, 255), 2 * th)
        cv2.arrowedLine(vis, tuple(int(v) for v in o), tuple(int(v) for v in x), (0, 0, 255), 2 * th, tipLength=0.05)
        color = (0, 200, 0)
    else:
        color = (0, 0, 255)
    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), color, 4 * th)
    if msg:
        cv2.putText(vis, msg, (10, 30 * th), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * th, color, 2 * th)
    return vis
