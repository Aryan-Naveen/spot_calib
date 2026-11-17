"""Camera intrinsic models (pinhole, Brown-Conrady, Kannala-Brandt) with project/undistort."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CameraModel:
    name: str
    width: int
    height: int
    K: np.ndarray
    model: str = "pinhole"  # pinhole | brown | kb
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    frame: str = ""  # sensor (optical) frame name

    def project(self, P_cam: np.ndarray) -> np.ndarray:
        """Project Nx3 points in the camera optical frame to Nx2 pixels."""
        P = np.ascontiguousarray(P_cam, dtype=np.float64).reshape(-1, 1, 3)
        z = np.zeros(3)
        if self.model == "kb":
            uv, _ = cv2.fisheye.projectPoints(P, z, z, self.K, self.dist[:4])
        else:
            uv, _ = cv2.projectPoints(P, z, z, self.K, self.dist)
        return uv.reshape(-1, 2)

    def undistort_norm(self, px: np.ndarray) -> np.ndarray:
        """Nx2 pixels -> Nx2 normalized (undistorted) image coordinates."""
        p = np.ascontiguousarray(px, dtype=np.float64).reshape(-1, 1, 2)
        if self.model == "kb":
            n = cv2.fisheye.undistortPoints(p, self.K, self.dist[:4])
        elif not np.any(self.dist):
            n = cv2.undistortPoints(p, self.K, None)
        else:
            n = cv2.undistortPoints(p, self.K, self.dist)
        return n.reshape(-1, 2)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    def to_dict(self) -> dict:
        return {"name": self.name, "width": self.width, "height": self.height, "K": self.K.tolist(),
                "model": self.model, "dist": np.asarray(self.dist).tolist(), "frame": self.frame}

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        return cls(name=d["name"], width=d["width"], height=d["height"], K=np.array(d["K"], float),
                   model=d.get("model", "pinhole"), dist=np.array(d.get("dist", [0] * 5), float),
                   frame=d.get("frame", ""))


def pinhole_K(fx, fy, cx, cy, skew=0.0) -> np.ndarray:
    return np.array([[fx, skew, cx], [0, fy, cy], [0, 0, 1.0]])
