import numpy as np

from spot_calib.cam_zed import solver
from spot_calib.cam_zed.board import Board, detect
from spot_calib.cam_zed.mock import MockRig
from spot_calib.common import transforms as tf


def _collect(rig, board, n=14):
    samples = []
    t = 0.5
    k = 0
    while len(samples) < n and k < 80:
        corners = {}
        for c in ["zed", "frontleft", "frontright"]:
            uv, _ = detect(rig.render(c, t), board)
            if uv is not None:
                corners[c] = uv
        if "zed" in corners and len(corners) > 1:
            samples.append(solver.Sample(id=len(samples), corners=corners,
                                         body_T_cam={c: rig.body_T[c] for c in corners if c != "zed"},
                                         parent_T_cam={c: rig.parent_T[c] for c in corners if c != "zed"},
                                         status="approved"))
        t += rig.period
        k += 1
    return samples


def test_canonical_ordering_is_consistent_under_flip():
    board = Board()
    rig = MockRig(board, seed=1)
    img = rig.render("zed", 0.5)
    uv, msg = detect(img, board)
    assert uv is not None, msg
    from spot_calib.cam_zed.board import canonicalize
    flipped = uv[::-1].copy()  # 180 deg reversed ordering as OpenCV might return
    uv2, _ = canonicalize(img, flipped, board)
    assert np.allclose(uv, uv2)
    mirrored = uv.reshape(board.rows, board.cols, 2)[::-1].reshape(-1, 2).copy()
    uv3, _ = canonicalize(img, mirrored, board)
    assert np.allclose(uv, uv3)


def test_recovers_extrinsics():
    board = Board()
    rig = MockRig(board, seed=3)
    samples = _collect(rig, board)
    assert len(samples) >= 8, f"only {len(samples)} usable synthetic samples"
    obj = board.object_points()
    truth = rig.truth()
    for mode in ["independent", "joint"]:
        sol = solver.solve(samples, rig.models, obj, mode)
        for c, T in sol.S_T_Z.items():
            dt, dr = tf.pose_diff(T, truth[c])
            assert dt < 2e-3 and dr < 0.1, (mode, c, dt, dr)
        assert sol.rms["zed"] < 0.5
    sol = solver.solve(samples, rig.models, obj, "independent")
    agree = solver.cross_camera_agreement(sol, samples)
    assert agree is not None and agree[0] < 3e-3
    loo = solver.leave_one_out(samples, rig.models, obj, "independent",
                               solver.per_sample_estimates(samples, rig.models, obj))
    assert all(v["t_m"] < 5e-3 for v in loo.values())
