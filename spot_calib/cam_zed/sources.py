"""Image sources: ZED (pyzed) and Spot body cameras (bosdyn). No ROS."""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from .camera import CameraModel, pinhole_K

log = logging.getLogger(__name__)


class Frame:
    __slots__ = ("img", "ts", "idx")

    def __init__(self, img, ts, idx):
        self.img, self.ts, self.idx = img, ts, idx


class ZedSource:
    """Rectified left image of a ZED, grabbed continuously in a background thread."""

    def __init__(self, resolution: str = "HD1080", fps: int = 15, serial: int | None = None):
        import pyzed.sl as sl

        self.sl = sl
        self.cam = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, resolution)
        init.camera_fps = fps
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.coordinate_units = sl.UNIT.METER
        if serial:
            init.set_from_serial_number(serial)
        err = self.cam.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {err}")
        info = self.cam.get_camera_information()
        conf = info.camera_configuration
        calib = conf.calibration_parameters
        lc = calib.left_cam
        self.model = CameraModel("zed", conf.resolution.width, conf.resolution.height,
                                 pinhole_K(lc.fx, lc.fy, lc.cx, lc.cy), "pinhole", np.zeros(5), "zed_left_optical")
        try:
            st = calib.stereo_transform
            stereo_t = list(st.get_translation().get())
        except Exception:
            stereo_t = None
        try:
            baseline = float(calib.get_camera_baseline())
        except Exception:
            baseline = abs(stereo_t[0]) if stereo_t else None
        if baseline is not None and baseline > 1.0:  # SDK returned mm
            baseline /= 1000.0
        self.info = {"serial": int(info.serial_number), "model": str(info.camera_model),
                     "resolution": [conf.resolution.width, conf.resolution.height],
                     "baseline_m": baseline, "stereo_translation": stereo_t}
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        sl = self.sl
        rt = sl.RuntimeParameters()
        mat = sl.Mat()
        idx = 0
        while not self._stop:
            if self.cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
                time.sleep(0.01)
                continue
            self.cam.retrieve_image(mat, sl.VIEW.LEFT)
            img = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2BGR)
            ts = self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds() * 1e-9
            idx += 1
            with self._lock:
                self._frame = Frame(img, ts, idx)

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frame

    def fresh(self, after_idx: int, timeout: float = 2.0) -> Frame | None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            f = self.latest()
            if f is not None and f.idx > after_idx:
                return f
            time.sleep(0.005)
        return None

    def close(self):
        self._stop = True
        self._thread.join(timeout=2)
        self.cam.close()


def _camera_model_from_source(name: str, src, frame: str) -> CameraModel:
    which = src.WhichOneof("camera_models")
    if which == "pinhole":
        i = src.pinhole.intrinsics
        K = pinhole_K(i.focal_length.x, i.focal_length.y, i.principal_point.x, i.principal_point.y, i.skew.x)
        return CameraModel(name, src.cols, src.rows, K, "pinhole", np.zeros(5), frame)
    if which == "pinhole_brown_conrady":
        i = src.pinhole_brown_conrady.intrinsics
        p = i.pinhole_intrinsics
        K = pinhole_K(p.focal_length.x, p.focal_length.y, p.principal_point.x, p.principal_point.y, p.skew.x)
        return CameraModel(name, src.cols, src.rows, K, "brown", np.array([i.k1, i.k2, i.p1, i.p2, i.k3]), frame)
    if which == "kannala_brandt":
        i = src.kannala_brandt.intrinsics
        p = i.pinhole_intrinsics
        K = pinhole_K(p.focal_length.x, p.focal_length.y, p.principal_point.x, p.principal_point.y, p.skew.x)
        return CameraModel(name, src.cols, src.rows, K, "kb", np.array([i.k1, i.k2, i.k3, i.k4, 0.0]), frame)
    raise RuntimeError(f"{name}: image source has no camera model")


class SpotSource:
    """Spot body cameras, e.g. cams=["frontleft", "frontright"] -> "<cam>_fisheye_image"."""

    def __init__(self, robot, cams: list[str], preview_hz: float = 2.0):
        from bosdyn.api import image_pb2
        from bosdyn.client.image import ImageClient, build_image_request

        self.cams = cams
        self.client = robot.ensure_client(ImageClient.default_service_name)
        self.requests = [build_image_request(f"{c}_fisheye_image", quality_percent=100,
                                             image_format=image_pb2.Image.FORMAT_RAW,
                                             pixel_format=image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8) for c in cams]
        self._image_pb2 = image_pb2
        self.models: dict[str, CameraModel] = {}
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._stop = False
        self._period = 1.0 / preview_hz
        self.capture()  # fail fast + fill models
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _decode(self, img_pb) -> np.ndarray:
        I = self._image_pb2.Image
        if img_pb.format == I.FORMAT_JPEG:
            return cv2.imdecode(np.frombuffer(img_pb.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        a = np.frombuffer(img_pb.data, np.uint8)
        if img_pb.pixel_format == I.PIXEL_FORMAT_GREYSCALE_U8:
            return a.reshape(img_pb.rows, img_pb.cols)
        if img_pb.pixel_format == I.PIXEL_FORMAT_RGB_U8:
            return cv2.cvtColor(a.reshape(img_pb.rows, img_pb.cols, 3), cv2.COLOR_RGB2GRAY)
        raise RuntimeError(f"unsupported pixel format {img_pb.pixel_format}")

    def capture(self) -> dict:
        """Synchronous capture of all cameras -> {cam: {img, model, body_T_cam, parent_T_cam, ts}}."""
        from bosdyn.client.frame_helpers import BODY_FRAME_NAME, get_a_tform_b

        resps = self.client.get_image(self.requests)
        out = {}
        for cam, r in zip(self.cams, resps):
            shot = r.shot
            sensor = shot.frame_name_image_sensor
            model = _camera_model_from_source(cam, r.source, sensor)
            self.models[cam] = model
            snap = shot.transforms_snapshot
            body_T = get_a_tform_b(snap, BODY_FRAME_NAME, sensor)
            try:
                parent_T = get_a_tform_b(snap, cam, sensor)
            except Exception:
                parent_T = None
            out[cam] = {"img": self._decode(shot.image), "model": model,
                        "body_T_cam": body_T.to_matrix() if body_T is not None else None,
                        "parent_T_cam": parent_T.to_matrix() if parent_T is not None else None,
                        "ts": shot.acquisition_time.seconds + shot.acquisition_time.nanos * 1e-9}
        return out

    def _loop(self):
        while not self._stop:
            t0 = time.time()
            try:
                cap = self.capture()
                with self._lock:
                    self._latest = cap
            except Exception as e:
                log.warning("Spot preview capture failed: %s", e)
            time.sleep(max(0.0, self._period - (time.time() - t0)))

    def latest(self) -> dict | None:
        with self._lock:
            return self._latest

    def close(self):
        self._stop = True
