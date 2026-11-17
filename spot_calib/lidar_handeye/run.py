"""Offline LiDAR <- body hand-eye calibration from a ROS 2 bag."""

from __future__ import annotations

import base64
import html
import logging
from pathlib import Path

import cv2
import numpy as np

from ..common import export, webapp
from ..common import transforms as tf
from . import registration, solver
from .stations import Station, build_cloud, detect_stations

log = logging.getLogger(__name__)

THRESHOLDS = {"min_stations": 12, "axis_coverage_deg": 10.0, "min_trans_singular": 0.15, "median_rot_res_deg": 0.2,
              "median_trans_res_mm": 10.0, "loo_t_mm": 10.0, "loo_r_deg": 0.3, "min_fitness": 0.5}


def make_pairs(n: int, mode: str) -> list[tuple[int, int]]:
    if mode == "all":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    s = set()
    for j in range(1, n):
        s.add((j - 1, j))
        if j >= 2:
            s.add((j - 2, j))
        s.add((0, j))
    return sorted(s)


def register_pairs(stations: list[Station], idx_pairs, X: np.ndarray, backend: str, min_fitness: float) -> list[solver.Pair]:
    by_id = {s.id: s for s in stations}
    out = []
    Xi = tf.inv(X)
    for k, (i, j) in enumerate(idx_pairs):
        si, sj = by_id[i], by_id[j]
        A = tf.inv(si.odom_T_body) @ sj.odom_T_body
        r = registration.register(si.points, sj.points, Xi @ A @ X, backend)
        ok = r.fitness >= min_fitness
        out.append(solver.Pair(i, j, A, r.T, r.fitness, r.rmse, ok))
        log.info("pair %2d-%2d  fitness %.2f  rmse %.3f m  %s", i, j, r.fitness, r.rmse, "" if ok else "REJECTED")
    return out


def _raster(clouds_colors, axes=(0, 1), extent=8.0, px=360) -> np.ndarray:
    img = np.full((px, px, 3), 255, np.uint8)
    for P, color in clouds_colors:
        u = ((P[:, axes[0]] + extent) / (2 * extent) * px).astype(int)
        v = (px - 1 - (P[:, axes[1]] + extent) / (2 * extent) * px).astype(int)
        m = (u >= 0) & (u < px) & (v >= 0) & (v < px)
        img[v[m], u[m]] = color
    return img


def pair_image(si: Station, sj: Station, B: np.ndarray) -> str:
    src = tf.transform_points(B, sj.points)
    top = _raster([(si.points, (150, 150, 150)), (src, (40, 40, 220))], (0, 1))
    side = _raster([(si.points, (150, 150, 150)), (src, (40, 40, 220))], (0, 2))
    img = np.hstack([top, np.full((top.shape[0], 4, 3), 0, np.uint8), side])
    return base64.b64encode(webapp.encode_jpeg(img, 80)).decode()


def badges(n_stations, obs, res, loo, th) -> list[dict]:
    cov = obs["axis_coverage_deg"]
    b = [webapp.badge("n", "Stations used", n_stations, th["min_stations"], ">=", "", "{:d}")]
    for k, name in enumerate(["roll", "pitch", "yaw"]):
        b.append(webapp.badge(f"cov_{name}", f"{name} excitation", cov[k], th["axis_coverage_deg"], ">=", "deg", "{:.1f}"))
    b += [webapp.badge("obs_t", "Translation observability (min σ)", obs["translation_singular_values"][-1],
                       th["min_trans_singular"], ">=", "", "{:.3f}"),
          webapp.badge("res_r", "Median rotation residual", float(np.median(res.res_rot_deg)), th["median_rot_res_deg"], "<", "deg"),
          webapp.badge("res_t", "Median translation residual", float(np.median(res.res_trans_m)) * 1e3, th["median_trans_res_mm"], "<", "mm"),
          webapp.badge("loo_t", "Leave-one-out σ translation", loo.get("t_m", np.nan) * 1e3 if loo else None, th["loo_t_mm"], "<", "mm"),
          webapp.badge("loo_r", "Leave-one-out σ rotation", loo.get("r_deg") if loo else None, th["loo_r_deg"], "<", "deg")]
    return b


def save_stations(path: Path, stations: list[Station]):
    np.savez_compressed(path, **{f"pts_{s.id}": s.points for s in stations},
                        meta=np.array([[s.id, s.t0, s.t1, s.spread_t, s.spread_r, s.n_frames] for s in stations]),
                        poses=np.stack([s.odom_T_body for s in stations]))


def load_stations(path: Path) -> list[Station]:
    d = np.load(path)
    out = []
    for row, T in zip(d["meta"], d["poses"]):
        sid = int(row[0])
        out.append(Station(sid, row[1], row[2], T, row[3], row[4], d[f"pts_{sid}"], int(row[5])))
    return out


def run(args) -> dict:
    robot = args.robot_name
    rn = "$(var robot_name)"
    frame_id, child = f"{rn}/body", f"{rn}/livox"
    X_prior = export.load_existing(args.calib_yaml, frame_id, child)
    prior_src = args.calib_yaml if X_prior is not None else "identity"
    if X_prior is None:
        X_prior = np.eye(4)
    session = export.new_session_dir(args.out, "lidar_handeye")
    th = dict(THRESHOLDS)
    th["min_fitness"] = args.min_fitness

    if args.from_session:
        stations = load_stations(Path(args.from_session) / "stations.npz")
        log.info("loaded %d stations from %s", len(stations), args.from_session)
        bag_info = export.load_json(Path(args.from_session) / "report.json").get("bag")
    else:
        from .bag import read_lidar_windows, read_odom

        lidar_topic = args.lidar_topic or f"/{robot}/livox/lidar"
        odom_topic = args.odom_topic or f"/{robot}/odom"
        odom = read_odom(args.bag, odom_topic)
        stations = detect_stations(odom, args.still_lin, args.still_ang, args.min_still)
        log.info("found %d stationary windows", len(stations))
        frames = read_lidar_windows(args.bag, lidar_topic, [(s.t0, s.t1) for s in stations], args.max_frames)
        for s, fr in zip(stations, frames):
            s.points = build_cloud(fr, args.min_range, args.max_range, args.voxel)
            s.n_frames = len(fr)
        stations = [s for s in stations if s.n_frames > 0 and len(s.points) > 500]
        bag_info = {"path": str(args.bag), "lidar_topic": lidar_topic, "odom_topic": odom_topic,
                    "odom_frame": odom.frame_id, "odom_child": odom.child_frame_id}
    save_stations(session / "stations.npz", stations)
    excluded = {int(x) for x in args.exclude.split(",") if x.strip()} if args.exclude else set()
    used = [s for s in stations if s.id not in excluded]
    if len(used) < 3:
        raise RuntimeError(f"only {len(used)} usable stations; need >= 3 (ideally {th['min_stations']}+)")
    ids = [s.id for s in used]
    idx_pairs = [(ids[a], ids[b]) for a, b in make_pairs(len(ids), args.pairs)]

    # Round 1 with the prior as registration seed, round 2 re-registers seeded by the estimate.
    X = X_prior
    pairs: list[solver.Pair] = []
    for rnd in range(2):
        log.info("registration round %d (%d pairs, backend %s)", rnd + 1, len(idx_pairs), args.backend)
        pairs = register_pairs(used, idx_pairs, X, args.backend, args.min_fitness)
        X = solver.solve(pairs, X_prior).X
    res = solver.solve(pairs, X_prior)
    X = res.X
    X_map = None
    if args.map_refine:
        X_map = solver.map_refine(X, pairs, {s.id: s.points for s in used})
        dt, dr = tf.pose_diff(X, X_map)
        log.info("map refinement moved X by %.1f mm / %.2f deg", dt * 1e3, dr)
    obs = solver.observability(pairs)
    loo = solver.leave_one_station_out(pairs, ids, X_prior)
    bdg = badges(len(used), obs, res, loo, th)
    X_out = X_map if X_map is not None else X
    entry = export.format_entry(frame_id, child, X_out)
    comp = export.compare(X_out, export.load_existing(args.calib_yaml, frame_id, child))

    ok_pairs = [p for p in pairs if p.ok]
    pair_rows = []
    for p in pairs:
        rr, rt = solver.pair_residual(p, X_out)
        pair_rows.append({"i": p.i, "j": p.j, "fitness": p.fitness, "rmse_m": p.rmse, "ok": p.ok,
                          "motion_deg": tf.rot_angle_deg(p.A[:3, :3]), "motion_m": float(np.linalg.norm(p.A[:3, 3])),
                          "res_rot_deg": rr, "res_trans_mm": rt * 1e3})
    report = {
        "bag": bag_info, "session": str(session), "frame_id": frame_id, "child_frame_id": child,
        "body_T_livox": tf.describe(X_out), "body_T_livox_matrix": X_out, "handeye_only": tf.describe(X),
        "map_refined": tf.describe(X_map) if X_map is not None else None,
        "sigma_t_m": res.sigma_t, "sigma_r_deg": res.sigma_r, "loo": loo, "init": res.init,
        "prior": tf.describe(X_prior), "prior_source": prior_src, "observability": obs, "badges": bdg,
        "comparison_with_existing": comp, "excluded": sorted(excluded), "thresholds": th,
        "stations": [{"id": s.id, "t0": s.t0, "t1": s.t1, "frames": s.n_frames, "points": len(s.points),
                      "spread_mm": s.spread_t * 1e3, "spread_deg": s.spread_r, "excluded": s.id in excluded,
                      "pose": tf.describe(s.odom_T_body)} for s in stations],
        "pairs": pair_rows, "backend": args.backend if args.backend != "auto" else (
            "small_gicp" if registration.HAVE_SMALL_GICP else "scipy"),
    }
    (session / "calibration_entry.yaml").write_text(entry)
    export.dump_json(session / "report.json", report)
    export.dump_json(session / "pairs.json", [p.to_dict() for p in pairs])
    by_id = {s.id: s for s in stations}
    imgs = {(p.i, p.j): pair_image(by_id[p.i], by_id[p.j], p.B) for p in pairs[:60]}
    (session / "report.html").write_text(render_html(report, entry, imgs))
    report["yaml"] = entry
    return report


def render_html(r: dict, entry: str, imgs: dict) -> str:
    e = html.escape
    badges_html = "".join(
        f'<div class="badge {"ok" if b["ok"] else ""}"><div class="t">{e(b["label"])}</div>'
        f'<div class="v">{e(b["value"])} <span class="t">{e(b["unit"])}</span></div><div class="t">target {e(b["threshold"])}</div></div>'
        for b in r["badges"])
    all_ok = all(b["ok"] for b in r["badges"])
    st_rows = "".join(
        f'<tr class="{"rejected" if s["excluded"] else ""}"><td>{s["id"]}</td><td>{s["t1"] - s["t0"]:.1f}s</td><td>{s["frames"]}</td>'
        f'<td>{s["points"]}</td><td>{s["spread_mm"]:.1f}</td><td>{s["spread_deg"]:.2f}</td>'
        f'<td>{", ".join(f"{v:.1f}" for v in s["pose"]["rpy_deg"])}</td></tr>' for s in r["stations"])
    pr_rows = "".join(
        f'<tr class="{"" if p["ok"] else "rejected"}"><td>{p["i"]}–{p["j"]}</td><td>{p["motion_deg"]:.1f}° / {p["motion_m"]:.2f} m</td>'
        f'<td>{p["fitness"]:.2f}</td><td>{p["rmse_m"] * 100:.1f}</td><td>{p["res_rot_deg"]:.3f}</td><td>{p["res_trans_mm"]:.1f}</td></tr>'
        for p in r["pairs"])
    pics = "".join(f'<figure><img class="thumb" src="data:image/jpeg;base64,{b}"><figcaption>stations {i}–{j}: '
                   f'top (x-y) | side (x-z); grey = station {i}, blue = station {j} registered</figcaption></figure>'
                   for (i, j), b in imgs.items())
    comp = r["comparison_with_existing"]
    comp_html = (f"Δt {comp['delta_translation_m'] * 1e3:.1f} mm, ΔR {comp['delta_rotation_deg']:.2f}° vs existing entry"
                 if comp else "no existing body→livox entry found")
    hints = "".join(f"<div class='msg'>{e(h)}</div>" for h in r["observability"]["hints"])
    css = (Path(__file__).parents[1] / "common" / "static" / "common.css").read_text()
    d = r["body_T_livox"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LiDAR Hand-Eye Report</title><style>{css}</style></head><body>
<header><h1>LiDAR ↔ body hand-eye</h1><span class="sub">{e(str(r["bag"]))}</span><span style="flex:1"></span>
<span class="alldone {"ok" if all_ok else "no"}">{"All targets met" if all_ok else f"{sum(b['ok'] for b in r['badges'])}/{len(r['badges'])} targets met"}</span></header>
<main>
<section class="panel"><h2>Result</h2><div class="kv">
<span class="muted">xyz [m]</span><span>{", ".join(f"{v:.4f}" for v in d["xyz"])}</span>
<span class="muted">rpy [deg]</span><span>{", ".join(f"{v:.3f}" for v in d["rpy_deg"])}</span>
<span class="muted">1σ (Gauss-Newton)</span><span>{r["sigma_t_m"] * 1e3:.1f} mm, {r["sigma_r_deg"]:.3f}°</span>
<span class="muted">compare</span><span>{e(comp_html)}</span>
<span class="muted">registration</span><span>{e(r["backend"])}</span>
<span class="muted">session</span><span>{e(r["session"])}</span></div>
<p class="muted">Paste into <code>platforms/&lt;platform&gt;/calibration.yaml</code>, replacing the existing <code>…/livox</code> entry:</p>
<textarea readonly>{e(entry)}</textarea></section>
<section class="panel"><h2>Accuracy</h2>{hints}<div class="badges">{badges_html}</div></section>
<section class="panel"><h2>Stations</h2><table><thead><tr><th>#</th><th>window</th><th>scans</th><th>points</th><th>jitter mm</th><th>jitter °</th><th>body rpy °</th></tr></thead><tbody>{st_rows}</tbody></table>
<p class="muted">Re-run with <code>--from-session {e(r["session"])} --exclude 3,7</code> to drop stations without re-reading the bag.</p></section>
<section class="panel"><h2>Pairs</h2><table><thead><tr><th>pair</th><th>body motion</th><th>fitness</th><th>rmse cm</th><th>res °</th><th>res mm</th></tr></thead><tbody>{pr_rows}</tbody></table></section>
<section class="panel"><h2>Registrations</h2><div class="grid-streams">{pics}</div></section>
</main></body></html>"""
