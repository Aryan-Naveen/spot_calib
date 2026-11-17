"""cam-zed: web GUI to collect checkerboard views and solve Spot camera <- ZED extrinsics."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import jsonify, request

from ..common import export, webapp
from ..common import transforms as tf
from . import solver
from .board import Board, detect, draw
from .camera import CameraModel
from .solver import ZED, Sample

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"

DEFAULT_THRESHOLDS = {"min_samples": 15, "zed_rms_px": 0.5, "spot_rms_px": 1.0, "loo_t_mm": 3.0, "loo_r_deg": 0.2,
                      "agree_t_mm": 10.0, "agree_r_deg": 0.5, "max_board_motion_px": 1.0}


class Calibrator:
    def __init__(self, zed, spot, board: Board, robot_name: str, out_root: str, calib_yaml: str | None,
                 link_T_opt: np.ndarray, link_T_opt_source: str):
        self.zed, self.spot, self.board = zed, spot, board
        self.robot_name = robot_name
        self.obj = board.object_points()
        self.calib_yaml = calib_yaml
        self.link_T_opt = link_T_opt  # zed_camera_link_T_zed_left_optical
        self.link_T_opt_source = link_T_opt_source
        self.session = export.new_session_dir(out_root, "cam_zed")
        (self.session / "images").mkdir(exist_ok=True)
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        self.samples: list[Sample] = []
        self.pending: Sample | None = None
        self.pending_info: dict = {}
        self.thumbs: dict[tuple[int, str], bytes] = {}
        self.raw: dict[tuple[int, str], np.ndarray] = {}
        self.lock = threading.RLock()
        self.capture_lock = threading.Lock()
        self.result: dict = {}
        self.solving = False
        self._dirty = threading.Event()
        self.previews: dict[str, bytes] = {}
        self.models: dict[str, CameraModel] = {ZED: zed.model, **spot.models}
        threading.Thread(target=self._solver_loop, daemon=True).start()
        threading.Thread(target=self._preview_loop, args=(ZED,), daemon=True).start()
        for c in spot.cams:
            threading.Thread(target=self._preview_loop, args=(c,), daemon=True).start()
        self._save_cameras()
        log.info("Session dir: %s", self.session)

    # ---------------------------------------------------------------- preview
    def _preview_loop(self, cam: str):
        while True:
            t0 = time.time()
            try:
                if cam == ZED:
                    f = self.zed.latest()
                    img = f.img if f is not None else None
                else:
                    cap = self.spot.latest()
                    img = cap[cam]["img"] if cap and cam in cap else None
                if img is None:
                    self.previews[cam] = webapp.encode_jpeg(webapp.placeholder(f"waiting for {cam}"))
                else:
                    small = webapp.downscale(img, 960)
                    uv, msg = detect(small, self.board, fast=True)
                    self.previews[cam] = webapp.encode_jpeg(draw(small, uv, self.board, msg or "board OK"), 70)
            except Exception as e:  # keep the stream alive
                log.warning("preview %s: %s", cam, e)
            time.sleep(max(0.05, 0.25 - (time.time() - t0)))

    # ---------------------------------------------------------------- capture
    def capture(self) -> dict:
        with self.capture_lock:
            f1 = self.zed.latest()
            if f1 is None:
                raise RuntimeError("no ZED frame yet")
            spot = self.spot.capture()
            f2 = self.zed.fresh(f1.idx)
            if f2 is None:
                raise RuntimeError("ZED stopped streaming")
            c1, m1 = detect(f1.img, self.board)
            c2, m2 = detect(f2.img, self.board)
            if c1 is None or c2 is None:
                raise RuntimeError(f"ZED: {m1 or m2}")
            motion = float(np.max(np.linalg.norm(c1 - c2, axis=1)))
            if motion > self.thresholds["max_board_motion_px"]:
                raise RuntimeError(f"board moved {motion:.1f}px during capture — hold it still and retry")
            corners = {ZED: 0.5 * (c1 + c2)}
            imgs = {ZED: f2.img}
            msgs = {}
            for c, d in spot.items():
                self.models[c] = d["model"]
                uv, msg = detect(d["img"], self.board)
                imgs[c] = d["img"]
                if uv is not None:
                    corners[c] = uv
                else:
                    msgs[c] = msg
            if len(corners) < 2:
                raise RuntimeError("no Spot camera sees the full board: " + "; ".join(f"{k}: {v}" for k, v in msgs.items()))
            with self.lock:
                sid = (max([s.id for s in self.samples]) + 1) if self.samples else 1
                if self.pending is not None:
                    sid = max(sid, self.pending.id + 1)
                s = Sample(id=sid, corners=corners,
                           body_T_cam={c: spot[c]["body_T_cam"] for c in corners if c != ZED and spot[c]["body_T_cam"] is not None},
                           parent_T_cam={c: spot[c]["parent_T_cam"] for c in corners if c != ZED and spot[c]["parent_T_cam"] is not None})
                for c, img in imgs.items():
                    self.raw[(sid, c)] = img
                    self.thumbs[(sid, c)] = webapp.encode_jpeg(draw(webapp.downscale(img, 960) if c == ZED else img,
                                                                    (corners[c] * (min(1.0, 960 / img.shape[1]) if c == ZED else 1.0)) if c in corners else None,
                                                                    self.board, msgs.get(c, "")), 75)
                self.pending = s
                self.pending_info = self._evaluate_pending(s, msgs, motion)
            return self.pending_info

    def _evaluate_pending(self, s: Sample, msgs: dict, motion: float) -> dict:
        est = solver.per_sample_estimates([s], self.models, self.obj)[s.id]
        info = {"id": s.id, "cams": sorted(s.corners), "missing": msgs, "board_motion_px": motion,
                "pnp_rms": est["pnp_rms"], "deviation": {}, "whatif": None}
        approved = self.approved()
        cur = self.result.get("independent")
        if cur:
            for c, T in est["S_T_Z"].items():
                if c in cur["S_T_Z_T"]:
                    dt, dr = tf.pose_diff(np.array(cur["S_T_Z_T"][c]), T)
                    info["deviation"][c] = {"t_mm": dt * 1e3, "r_deg": dr}
        try:
            sol = solver.solve(approved + [s], self.models, self.obj, "independent")
            if sol:
                info["whatif"] = {"rms": sol.rms, "sample_rms": sol.sample_rms.get(s.id, {})}
        except Exception as e:
            log.warning("what-if solve failed: %s", e)
        return info

    # ---------------------------------------------------------------- review
    def approved(self) -> list[Sample]:
        return [s for s in self.samples if s.status == "approved"]

    def approve(self, sid: int):
        with self.lock:
            if self.pending is None or self.pending.id != sid:
                raise RuntimeError("no such pending sample")
            s = self.pending
            s.status = "approved"
            self.samples.append(s)
            self.pending = None
            for c in s.corners.keys() | {k[1] for k in self.raw if k[0] == sid}:
                if (sid, c) in self.raw:
                    cv2.imwrite(str(self.session / "images" / f"{sid:03d}_{c}.png"), self.raw.pop((sid, c)))
            self._save_samples()
        self._dirty.set()

    def reject(self, sid: int):
        with self.lock:
            if self.pending is not None and self.pending.id == sid:
                self.pending = None
                for k in [k for k in self.raw if k[0] == sid]:
                    self.raw.pop(k)
                return
            for s in self.samples:
                if s.id == sid:
                    s.status = "rejected" if s.status == "approved" else "approved"
            self._save_samples()
        self._dirty.set()

    # ---------------------------------------------------------------- solve
    def _solver_loop(self):
        while True:
            self._dirty.wait()
            self._dirty.clear()
            self.solving = True
            try:
                res = self.compute(self.approved())
                with self.lock:
                    self.result = res
            except Exception as e:
                log.exception("solve failed: %s", e)
            finally:
                self.solving = False

    def compute(self, samples: list[Sample]) -> dict:
        res: dict = {"n": len(samples)}
        if len(samples) < 2:
            return res
        init = solver.per_sample_estimates(samples, self.models, self.obj)
        for mode in ("independent", "joint"):
            sol = solver.solve(samples, self.models, self.obj, mode, init)
            if sol is None:
                continue
            d = sol.summary()
            d["S_T_Z_T"] = {c: T.tolist() for c, T in sol.S_T_Z.items()}
            d["body_T_Z_T"] = sol.body_T_Z.tolist() if sol.body_T_Z is not None else None
            d["sample_rms"] = {str(k): v for k, v in sol.sample_rms.items()}
            if mode == "independent":
                d["agreement"] = solver.cross_camera_agreement(sol, samples)
            res[mode] = d
        # per-sample deviation from final independent estimate
        if "independent" in res:
            dev = {}
            for s in samples:
                for c, T in init[s.id]["S_T_Z"].items():
                    if c in res["independent"]["S_T_Z_T"]:
                        dt, dr = tf.pose_diff(np.array(res["independent"]["S_T_Z_T"][c]), T)
                        dev.setdefault(str(s.id), {})[c] = {"t_mm": dt * 1e3, "r_deg": dr}
            res["deviation"] = dev
        res["loo"] = solver.leave_one_out(samples, self.models, self.obj, "independent", init) if len(samples) >= 4 else {}
        res["coverage"] = solver.coverage(samples, self.models)
        return res

    def badges(self) -> list[dict]:
        th, r = self.thresholds, self.result
        ind = r.get("independent", {})
        rms = ind.get("rms", {})
        spot_rms = max([v for k, v in rms.items() if k != ZED], default=None)
        loo = r.get("loo", {})
        loo_t = max([v["t_m"] * 1e3 for v in loo.values()], default=None)
        loo_r = max([v["r_deg"] for v in loo.values()], default=None)
        b = [webapp.badge("n", "Approved samples", len(self.approved()), th["min_samples"], ">=", "", "{:d}"),
             webapp.badge("zed_rms", "ZED reprojection RMS", rms.get(ZED), th["zed_rms_px"], "<", "px"),
             webapp.badge("spot_rms", "Spot reprojection RMS (worst)", spot_rms, th["spot_rms_px"], "<", "px"),
             webapp.badge("loo_t", "Leave-one-out σ translation", loo_t, th["loo_t_mm"], "<", "mm"),
             webapp.badge("loo_r", "Leave-one-out σ rotation", loo_r, th["loo_r_deg"], "<", "deg")]
        if len(self.spot.cams) > 1:
            ag = ind.get("agreement")
            b += [webapp.badge("agree_t", "frontleft vs frontright (body)", ag[0] * 1e3 if ag else None, th["agree_t_mm"], "<", "mm"),
                  webapp.badge("agree_r", "frontleft vs frontright (body)", ag[1] if ag else None, th["agree_r_deg"], "<", "deg")]
        return b

    # ---------------------------------------------------------------- export
    def _parent_T_S(self, cam: str) -> np.ndarray | None:
        Ts = [s.parent_T_cam[cam] for s in self.approved() if cam in s.parent_T_cam]
        return tf.average_poses(Ts) if Ts else None

    def _body_T_S(self, cam: str) -> np.ndarray | None:
        Ts = [s.body_T_cam[cam] for s in self.approved() if cam in s.body_T_cam]
        return tf.average_poses(Ts) if Ts else None

    def export(self, mode: str = "independent", parent: str = "frontleft") -> dict:
        r = self.result
        if mode not in r:
            raise RuntimeError("no solution yet")
        sol = r[mode]
        if parent not in sol["S_T_Z_T"]:
            raise RuntimeError(f"{parent} not in solution")
        S_T_Z = np.array(sol["S_T_Z_T"][parent])
        opt_T_link = tf.inv(self.link_T_opt)
        parent_T_S = self._parent_T_S(parent)
        body_T_S = self._body_T_S(parent)
        rn = "$(var robot_name)"
        child = f"{rn}_zed_camera_link"
        if parent_T_S is not None:
            frame_id = f"{rn}/{parent}"
            P_T_link = parent_T_S @ S_T_Z @ opt_T_link
        else:
            frame_id = f"{rn}/{self.models[parent].frame}"
            P_T_link = S_T_Z @ opt_T_link
        yaml_entry = export.format_entry(frame_id, child, P_T_link)
        old = export.load_existing(self.calib_yaml, frame_id, child)
        report = {
            "mode": mode, "parent": parent, "frame_id": frame_id, "child_frame_id": child,
            "parent_T_zed_camera_link": tf.describe(P_T_link),
            "sensor_T_zed_left_optical": {c: tf.describe(np.array(T)) for c, T in sol["S_T_Z_T"].items()},
            "body_T_zed_camera_link": tf.describe(body_T_S @ S_T_Z @ opt_T_link) if body_T_S is not None else None,
            "body_T_zed_left_optical": tf.describe(body_T_S @ S_T_Z) if body_T_S is not None else None,
            "zed_link_T_left_optical": tf.describe(self.link_T_opt), "zed_link_T_left_optical_source": self.link_T_opt_source,
            "comparison_with_existing": export.compare(P_T_link, old), "calib_yaml": self.calib_yaml,
            "metrics": {k: r.get(k) for k in ("independent", "joint", "loo", "n")},
            "badges": self.badges(), "thresholds": self.thresholds, "board": vars(self.board),
            "spot_parent_T_sensor": {c: tf.describe(T) for c in self.spot.cams if (T := self._parent_T_S(c)) is not None},
        }
        (self.session / "calibration_entry.yaml").write_text(yaml_entry)
        export.dump_json(self.session / "report.json", report)
        return {"yaml": yaml_entry, "report": report, "session": str(self.session)}

    # ---------------------------------------------------------------- persistence
    def _save_cameras(self):
        export.dump_json(self.session / "cameras.json", {
            "models": {k: m.to_dict() for k, m in self.models.items()}, "zed_info": getattr(self.zed, "info", {}),
            "board": vars(self.board), "zed_link_T_left_optical": self.link_T_opt,
            "zed_link_T_left_optical_source": self.link_T_opt_source, "robot_name": self.robot_name,
            "spot_cams": self.spot.cams, "calib_yaml": self.calib_yaml})

    def _save_samples(self):
        self._save_cameras()
        export.dump_json(self.session / "samples.json", [s.to_dict() for s in self.samples])

    # ---------------------------------------------------------------- state
    def state(self) -> dict:
        with self.lock:
            dev = self.result.get("deviation", {})
            srms = self.result.get("independent", {}).get("sample_rms", {})
            rows = []
            for s in self.samples:
                d = dev.get(str(s.id), {})
                rr = srms.get(str(s.id), {})
                rows.append({"id": s.id, "status": s.status, "cams": ",".join(sorted(s.corners)),
                             "rms_zed": rr.get(ZED), "rms_spot": max([v for k, v in rr.items() if k != ZED], default=None),
                             "dev_t_mm": max([v["t_mm"] for v in d.values()], default=None),
                             "dev_r_deg": max([v["r_deg"] for v in d.values()], default=None)})
            res = {k: v for k, v in self.result.items() if k not in ("deviation",)}
            return {"robot_name": self.robot_name, "cams": [ZED] + list(self.spot.cams), "session": str(self.session),
                    "board": vars(self.board), "thresholds": self.thresholds, "samples": rows,
                    "pending": self.pending_info if self.pending is not None else None, "result": res,
                    "badges": self.badges(), "solving": self.solving or self._dirty.is_set(),
                    "link_T_opt_source": self.link_T_opt_source}


def build_app(cal: Calibrator):
    app = webapp.create_app(STATIC)

    def err(e, code=400):
        return jsonify({"error": str(e)}), code

    @app.get("/api/state")
    def state():
        return jsonify(_jsonable(cal.state()))

    @app.get("/stream/<cam>")
    def stream(cam):
        return webapp.mjpeg_response(lambda: cal.previews.get(cam), fps=5)

    @app.post("/api/capture")
    def capture():
        try:
            return jsonify(_jsonable(cal.capture()))
        except Exception as e:
            return err(e)

    @app.post("/api/sample/<int:sid>/approve")
    def approve(sid):
        try:
            cal.approve(sid)
            return jsonify({"ok": True})
        except Exception as e:
            return err(e)

    @app.post("/api/sample/<int:sid>/reject")
    def reject(sid):
        cal.reject(sid)
        return jsonify({"ok": True})

    @app.get("/api/sample/<int:sid>/img/<cam>.jpg")
    def img(sid, cam):
        jpg = cal.thumbs.get((sid, cam))
        if jpg is None:
            return err("not found", 404)
        return app.response_class(jpg, mimetype="image/jpeg")

    @app.post("/api/thresholds")
    def thresholds():
        for k, v in (request.get_json() or {}).items():
            if k in cal.thresholds:
                cal.thresholds[k] = type(DEFAULT_THRESHOLDS[k])(v)
        return jsonify(cal.thresholds)

    @app.get("/api/export")
    def do_export():
        try:
            return jsonify(_jsonable(cal.export(request.args.get("mode", "independent"),
                                                request.args.get("parent", "frontleft"))))
        except Exception as e:
            return err(e)

    return app


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def resolve_session(session_dir: str, mode: str = "independent", parent: str = "frontleft") -> dict:
    """Re-solve a saved session offline (no hardware)."""
    d = Path(session_dir)
    cams = export.load_json(d / "cameras.json")
    samples = [Sample.from_dict(s) for s in export.load_json(d / "samples.json")]

    class _Stub:
        pass

    zed = _Stub()
    zed.model = CameraModel.from_dict(cams["models"][ZED])
    zed.info = cams.get("zed_info", {})
    spot = _Stub()
    spot.cams = cams["spot_cams"]
    spot.models = {k: CameraModel.from_dict(v) for k, v in cams["models"].items() if k != ZED}
    cal = Calibrator.__new__(Calibrator)
    cal.zed, cal.spot, cal.board = zed, spot, Board(**cams["board"])
    cal.robot_name = cams.get("robot_name", "")
    cal.obj = cal.board.object_points()
    cal.calib_yaml = cams.get("calib_yaml")
    cal.link_T_opt = np.array(cams["zed_link_T_left_optical"])
    cal.link_T_opt_source = cams.get("zed_link_T_left_optical_source", "")
    cal.session = d
    cal.thresholds = dict(DEFAULT_THRESHOLDS)
    cal.samples = samples
    cal.models = {ZED: zed.model, **spot.models}
    cal.lock = threading.RLock()
    cal.result = cal.compute(cal.approved())
    out = cal.export(mode, parent)
    out["badges"] = cal.badges()
    return out
