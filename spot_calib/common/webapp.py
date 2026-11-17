"""Shared Flask plumbing: app factory, MJPEG streaming, threshold badges."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, send_from_directory

COMMON_STATIC = Path(__file__).parent / "static"


def create_app(tool_static: Path) -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(tool_static, "index.html")

    @app.get("/static/<path:name>")
    def tool_static_file(name):
        return send_from_directory(tool_static, name)

    @app.get("/common/<path:name>")
    def common_static_file(name):
        return send_from_directory(COMMON_STATIC, name)

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    return app


def encode_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def placeholder(text: str, w: int = 640, h: int = 360) -> np.ndarray:
    img = np.full((h, w, 3), 40, np.uint8)
    cv2.putText(img, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    return img


def mjpeg_response(get_jpeg, fps: float = 5.0) -> Response:
    """Stream ``get_jpeg()`` (bytes or None) as multipart MJPEG."""

    def gen():
        last = None
        while True:
            jpg = get_jpeg()
            if jpg is not None and jpg is not last:
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(1.0 / fps)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def badge(key: str, label: str, value, threshold, op: str = "<", unit: str = "", fmt: str = "{:.3g}") -> dict:
    """A pass/fail metric shown in the GUI. ``value`` None means not yet computable."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        ok, shown = False, "—"
    else:
        ok = value < threshold if op == "<" else value >= threshold
        shown = fmt.format(value)
    return {"key": key, "label": label, "value": shown, "threshold": f"{op} {fmt.format(threshold)} {unit}".strip(),
            "unit": unit, "ok": bool(ok)}


def downscale(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    s = max_w / w
    return cv2.resize(img, (max_w, int(round(h * s))), interpolation=cv2.INTER_AREA)
