"""spot-calib command line."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _print_result(yaml_entry: str, badges: list[dict], extra: str = ""):
    print("\n=== Accuracy ===")
    for b in badges:
        print(f"  [{'OK ' if b['ok'] else ' X '}] {b['label']:<40} {b['value']:>10} {b['unit']:<4} (target {b['threshold']})")
    if extra:
        print(extra)
    print("\n=== calibration.yaml entry ===\n" + yaml_entry)


def cmd_cam_zed(a):
    import numpy as np

    from .cam_zed import zed_frames
    from .cam_zed.app import Calibrator, build_app, resolve_session
    from .cam_zed.board import Board

    if a.resolve:
        out = resolve_session(a.resolve, a.mode, a.parent)
        c = out["report"]["comparison_with_existing"]
        _print_result(out["yaml"], out["badges"], f"\nvs existing: Δt {c['delta_translation_m'] * 1e3:.1f} mm, "
                                                    f"ΔR {c['delta_rotation_deg']:.2f}°" if c else "")
        print(f"report: {out['session']}/report.json")
        return
    board = Board(a.cols, a.rows, a.square)
    cams = [c.strip() for c in a.spot_cams.split(",") if c.strip()]
    if a.mock:
        import time

        from .cam_zed.mock import MockRig, MockSpot, MockZed

        rig = MockRig(board)
        zed = MockZed(rig)
        spot = MockSpot(rig, cams, time.time())
        spot._t0 = zed._t0
        logging.info("MOCK mode. Ground truth sensor_T_zed_optical: %s",
                     {c: np.round(T, 4).tolist() for c, T in rig.truth().items()})
    else:
        from .cam_zed.sources import SpotSource, ZedSource
        from .common.spot_session import connect

        zed = ZedSource(a.zed_resolution, serial=a.zed_serial)
        spot = SpotSource(connect(a.spot_ip, a.spot_user, a.spot_password), cams)
    if a.zed_link_T_left_optical:
        link_T_opt = zed_frames.parse_override(a.zed_link_T_left_optical)
        src = "user override (--zed-link-T-left-optical)"
    else:
        link_T_opt = zed_frames.link_T_left_optical(a.zed_model, zed.info.get("baseline_m"))
        src = f"zed_wrapper {a.zed_model} URDF defaults (verified for zed2i)"
    cal = Calibrator(zed, spot, board, a.robot_name, a.out, a.calib_yaml, link_T_opt, src)
    app = build_app(cal)
    print(f"\n  Open http://<this-host>:{a.port}/ from any machine on the network.\n  Session: {cal.session}\n")
    app.run(host=a.host, port=a.port, threaded=True)


def cmd_lidar(a):
    from .lidar_handeye.run import run

    r = run(a)
    c = r["comparison_with_existing"]
    extra = "\n=== Observability hints ===\n" + "\n".join("  - " + h for h in r["observability"]["hints"]) if r["observability"]["hints"] else ""
    if c:
        extra += f"\nvs existing: Δt {c['delta_translation_m'] * 1e3:.1f} mm, ΔR {c['delta_rotation_deg']:.2f}°"
    _print_result(r["yaml"], r["badges"], extra)
    print(f"report: {r['session']}/report.html")


def cmd_mock_bag(a):
    from .lidar_handeye.mock import generate

    X = generate(a.out, a.robot_name)
    print(f"wrote {a.out}; ground-truth body_T_livox =\n{json.dumps(X.round(5).tolist())}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="spot-calib", description="Spot sensor extrinsic calibration")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cam-zed", help="checkerboard calibration of ZED vs Spot front cameras (web GUI)")
    c.add_argument("--robot-name", default="robot")
    c.add_argument("--calib-yaml", help="existing dcist platforms/<platform>/calibration.yaml (for comparison)")
    c.add_argument("--cols", type=int, default=10, help="inner corners per row")
    c.add_argument("--rows", type=int, default=7, help="inner corners per column")
    c.add_argument("--square", type=float, default=0.036, help="square size [m]")
    c.add_argument("--spot-cams", default="frontleft,frontright")
    c.add_argument("--zed-resolution", default="HD1080")
    c.add_argument("--zed-serial", type=int)
    c.add_argument("--zed-model", default="zed2i")
    c.add_argument("--zed-link-T-left-optical", help='"x y z qx qy qz qw" from tf2_echo <robot>_zed_camera_link '
                                                    '<robot>_zed_left_camera_frame_optical')
    c.add_argument("--spot-ip"), c.add_argument("--spot-user"), c.add_argument("--spot-password")
    c.add_argument("--host", default="0.0.0.0")
    c.add_argument("--port", type=int, default=8080)
    c.add_argument("--out", default="results")
    c.add_argument("--mock", action="store_true", help="synthetic cameras (no hardware)")
    c.add_argument("--resolve", metavar="SESSION_DIR", help="re-solve a saved session offline and print the entry")
    c.add_argument("--mode", default="independent", choices=["independent", "joint"])
    c.add_argument("--parent", default="frontleft")
    c.set_defaults(func=cmd_cam_zed)

    l = sub.add_parser("lidar-handeye", help="offline hand-eye calibration of the LiDAR vs body from a ROS 2 bag")
    l.add_argument("--bag", help="ROS 2 bag directory")
    l.add_argument("--from-session", metavar="SESSION_DIR", help="reuse stations extracted in a previous run")
    l.add_argument("--robot-name", default="robot")
    l.add_argument("--lidar-topic", help="default /<robot>/livox/lidar")
    l.add_argument("--odom-topic", help="default /<robot>/odom")
    l.add_argument("--calib-yaml", help="existing calibration.yaml: prior + comparison")
    l.add_argument("--exclude", help="comma-separated station ids to drop")
    l.add_argument("--pairs", default="chain", choices=["chain", "all"])
    l.add_argument("--backend", default="auto", choices=["auto", "small_gicp", "scipy"])
    l.add_argument("--map-refine", action="store_true", help="extra point-to-plane refinement of X over all stations")
    l.add_argument("--still-lin", type=float, default=0.02, help="max body speed for a station [m/s]")
    l.add_argument("--still-ang", type=float, default=0.02, help="max body angular rate for a station [rad/s]")
    l.add_argument("--min-still", type=float, default=1.5, help="min stationary duration [s]")
    l.add_argument("--max-frames", type=int, default=20, help="LiDAR scans accumulated per station")
    l.add_argument("--min-range", type=float, default=0.8, help="crop points closer than this (robot body) [m]")
    l.add_argument("--max-range", type=float, default=40.0)
    l.add_argument("--voxel", type=float, default=0.05)
    l.add_argument("--min-fitness", type=float, default=0.5)
    l.add_argument("--out", default="results")
    l.set_defaults(func=cmd_lidar)

    m = sub.add_parser("mock-bag", help="write a synthetic bag for trying lidar-handeye")
    m.add_argument("out")
    m.add_argument("--robot-name", default="mock")
    m.set_defaults(func=cmd_mock_bag)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.cmd == "lidar-handeye" and not (a.bag or a.from_session):
        p.error("lidar-handeye needs --bag or --from-session")
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
