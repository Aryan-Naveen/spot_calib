"""Synthetic ZED + Spot cameras rendering a checkerboard, for testing without hardware."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..common import transforms as tf
from .board import Board
from .camera import CameraModel, pinhole_K
from .sources import Frame


def _R(seq, deg):
    return Rotation.from_euler(seq, deg, degrees=True).as_matrix()


class MockRig:
    """Ground-truth geometry. Body frame: x forward, y left, z up."""

    def __init__(self, board: Board, seed: int = 0, period: float = 3.0):
        self.board = board
        self.period = period
        self.rng_seed = seed
        self.body_T_zed = tf.make_T(_R("y", 10), [0.45, 0.0, 0.15]) @ tf.FRAME_T_OPTICAL
        portrait = tf.make_T(_R("z", 90), [0, 0, 0])
        self.body_T = {
            "frontleft": tf.make_T(_R("zy", [-25, 15]), [0.42, 0.04, 0.0]) @ tf.FRAME_T_OPTICAL @ portrait,
            "frontright": tf.make_T(_R("zy", [25, 15]), [0.42, -0.04, 0.0]) @ tf.FRAME_T_OPTICAL @ portrait,
        }
        self.parent_T = {c: tf.make_T(_R("x", 3), [0.01, 0.0, 0.0]) for c in self.body_T}
        self.models = {
            "zed": CameraModel("zed", 1920, 1080, pinhole_K(1060, 1060, 960, 540), frame="zed_left_optical"),
            "frontleft": CameraModel("frontleft", 640, 480, pinhole_K(330, 330, 320, 240), frame="frontleft_fisheye"),
            "frontright": CameraModel("frontright", 640, 480, pinhole_K(330, 330, 320, 240), frame="frontright_fisheye"),
        }
        self.texture, self.tex_S = self._make_texture()

    def _make_texture(self, px: int = 40, margin: int = 40):
        b = self.board
        nx, ny = b.cols + 1, b.rows + 1
        img = np.full((ny * px + 2 * margin, nx * px + 2 * margin), 255, np.uint8)
        for j in range(ny):
            for i in range(nx):
                if (i + j) % 2 == 0:  # square (0,0) is black
                    img[margin + j * px:margin + (j + 1) * px, margin + i * px:margin + (i + 1) * px] = 0
        k = b.square / px
        o = margin + px
        S = np.array([[k, 0, -o * k], [0, k, -o * k], [0, 0, 1.0]])  # texture px -> board metres
        return img, S

    def body_T_board(self, t: float) -> np.ndarray:
        k = int(t // self.period)
        rng = np.random.default_rng(self.rng_seed * 100003 + k)
        b = self.board
        center = np.array([(b.cols - 1) * b.square / 2, (b.rows - 1) * b.square / 2, 0])
        # Board facing the robot: board z points along +x body (away from robot), board x ~ -y body.
        # columns = board x, y, z expressed in body
        R0 = np.stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]], axis=1).astype(float)
        tilt = _R("xyz", rng.uniform([-30, -30, -40], [30, 30, 40]))
        R = R0 @ tilt
        c = np.array([rng.uniform(1.1, 2.0), rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.15)])
        return tf.make_T(R, c - R @ center)

    def cam_T_board(self, cam: str, t: float) -> np.ndarray:
        bTc = self.body_T_zed if cam == "zed" else self.body_T[cam]
        return tf.inv(bTc) @ self.body_T_board(t)

    def render(self, cam: str, t: float, noise: float = 2.0) -> np.ndarray:
        m = self.models[cam]
        T = self.cam_T_board(cam, t)
        H = m.K @ np.column_stack([T[:3, 0], T[:3, 1], T[:3, 3]]) @ self.tex_S
        img = cv2.warpPerspective(self.texture, H, (m.width, m.height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=150)
        # board behind camera -> blank
        if T[2, 3] <= 0:
            img[:] = 150
        img = cv2.GaussianBlur(img, (3, 3), 0.7)
        rng = np.random.default_rng(int(t * 1000) % 2 ** 31)
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
        return img

    def truth(self) -> dict:
        return {c: tf.inv(T) @ self.body_T_zed for c, T in self.body_T.items()}


class MockZed:
    def __init__(self, rig: MockRig, fps: float = 10.0):
        self.rig = rig
        self.model = rig.models["zed"]
        self.info = {"serial": 0, "model": "mock", "resolution": [1920, 1080], "baseline_m": 0.12}
        self._lock = threading.Lock()
        self._frame = None
        self._t0 = time.time()
        self._stop = False
        self._fps = fps
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        idx = 0
        while not self._stop:
            t = time.time() - self._t0
            img = cv2.cvtColor(self.rig.render("zed", t), cv2.COLOR_GRAY2BGR)
            idx += 1
            with self._lock:
                self._frame = Frame(img, t, idx)
            time.sleep(1.0 / self._fps)

    def latest(self):
        with self._lock:
            return self._frame

    def fresh(self, after_idx, timeout=2.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            f = self.latest()
            if f is not None and f.idx > after_idx:
                return f
            time.sleep(0.005)
        return None

    def close(self):
        self._stop = True


class MockSpot:
    def __init__(self, rig: MockRig, cams: list[str], t0: float):
        self.rig, self.cams, self._t0 = rig, cams, t0
        self.models = {c: rig.models[c] for c in cams}

    def capture(self) -> dict:
        t = time.time() - self._t0
        return {c: {"img": self.rig.render(c, t), "model": self.rig.models[c], "body_T_cam": self.rig.body_T[c],
                    "parent_T_cam": self.rig.parent_T[c], "ts": t} for c in self.cams}

    def latest(self):
        return self.capture()

    def close(self):
        pass
