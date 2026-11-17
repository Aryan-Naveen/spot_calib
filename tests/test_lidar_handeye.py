import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spot_calib.common import transforms as tf
from spot_calib.lidar_handeye import registration, solver
from spot_calib.lidar_handeye.mock import X_TRUE, room_points, station_poses


def _pairs(poses, X, noise_t=0.002, noise_r_deg=0.05, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            A = tf.inv(poses[i]) @ poses[j]
            B = tf.inv(X) @ A @ X
            N = tf.make_T(Rotation.from_rotvec(np.radians(rng.normal(0, noise_r_deg, 3))).as_matrix(), rng.normal(0, noise_t, 3))
            out.append(solver.Pair(i, j, A, B @ N))
    return out


def test_handeye_recovers_X():
    pairs = _pairs(station_poses(), X_TRUE)
    res = solver.solve(pairs, np.eye(4))
    dt, dr = tf.pose_diff(res.X, X_TRUE)
    assert dt < 3e-3 and dr < 0.1, (dt, dr)
    obs = solver.observability(pairs)
    assert not obs["hints"]


def test_yaw_only_is_flagged():
    poses = [tf.make_T(Rotation.from_euler("z", y, degrees=True).as_matrix(), [0.3 * k, 0, 0.5])
             for k, y in enumerate([0, 30, 60, -40, 90])]
    obs = solver.observability(_pairs(poses, X_TRUE))
    assert obs["translation_singular_values"][-1] < 0.15
    assert abs(obs["weak_direction"][2]) > 0.9  # z is the unobservable direction
    assert any("roll" in h for h in obs["hints"])


def test_registration_recovers_motion():
    room = room_points(60_000, seed=1)
    rng = np.random.default_rng(2)
    T = tf.make_T(Rotation.from_euler("xyz", [5, -4, 20], degrees=True).as_matrix(), [0.3, -0.2, 0.05])
    tgt = room[rng.choice(len(room), 30_000, replace=False)]
    src = tf.transform_points(tf.inv(T), room[rng.choice(len(room), 30_000, replace=False)])
    r = registration.register(tgt, src, np.eye(4), backend="scipy")
    dt, dr = tf.pose_diff(r.T, T)
    assert dt < 0.01 and dr < 0.2, (dt, dr)
    assert r.fitness > 0.8


def test_bag_roundtrip(tmp_path):
    pytest.importorskip("rosbag2_py")
    from spot_calib.lidar_handeye.bag import read_odom
    from spot_calib.lidar_handeye.mock import generate
    from spot_calib.lidar_handeye.stations import detect_stations

    generate(str(tmp_path / "bag"), "mock")
    odom = read_odom(str(tmp_path / "bag"), "/mock/odom")
    st = detect_stations(odom)
    assert len(st) == len(station_poses())
    for s, P in zip(st, station_poses()):
        assert tf.pose_diff(s.odom_T_body, P)[0] < 1e-3
