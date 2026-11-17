"""Synthetic bag: Spot holding a series of body poses inside a room, with a Mid-360-like LiDAR."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..common import transforms as tf

X_TRUE = tf.make_T(Rotation.from_euler("xyz", [1.5, -2.0, 3.0], degrees=True).as_matrix(), [-0.205, 0.012, 0.462])


def room_points(n: int = 120_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    planes = []  # (origin, u, v) rectangles
    L, W, H = 12.0, 8.0, 3.2
    o = np.array([-4.0, -4.0, -0.5])
    ex, ey, ez = np.eye(3)
    planes += [(o, ex * L, ey * W), (o + ez * H, ex * L, ey * W),  # floor, ceiling
               (o, ex * L, ez * H), (o + ey * W, ex * L, ez * H), (o, ey * W, ez * H), (o + ex * L, ey * W, ez * H)]
    # boxes and a ramp for structure in every direction
    for c, s in [((2.0, 1.5, -0.5), (1.0, 0.8, 1.2)), ((-1.5, -2.0, -0.5), (0.6, 1.5, 0.9)), ((4.5, -1.0, -0.5), (0.5, 0.5, 2.0))]:
        c, s = np.array(c), np.array(s)
        planes += [(c + ez * s[2], ex * s[0], ey * s[1]), (c, ex * s[0], ez * s[2]), (c + ey * s[1], ex * s[0], ez * s[2]),
                   (c, ey * s[1], ez * s[2]), (c + ex * s[0], ey * s[1], ez * s[2])]
    R = Rotation.from_euler("y", -25, degrees=True).as_matrix()
    planes.append((np.array([0.0, 2.5, -0.5]), R @ ex * 2.0, ey * 1.2))
    areas = np.array([np.linalg.norm(np.cross(u, v)) for _, u, v in planes])
    counts = rng.multinomial(n, areas / areas.sum())
    pts = [o + rng.random((k, 1)) * u + rng.random((k, 1)) * v for (o, u, v), k in zip(planes, counts)]
    return np.concatenate(pts)


def station_poses() -> list[np.ndarray]:
    E = lambda r, p, y: Rotation.from_euler("xyz", [r, p, y], degrees=True).as_matrix()  # noqa: E731
    base_z = 0.5
    spec = [((0, 0, 0), (0, 0, 0)), ((15, 0, 0), (0, 0, -0.03)), ((-15, 0, 0), (0, 0, -0.03)),
            ((0, 12, 0), (0, 0, -0.05)), ((0, -12, 0), (0, 0, 0.0)), ((0, 0, 25), (0, 0, 0)),
            ((0, 0, -25), (0, 0, 0)), ((10, 8, 15), (0, 0, -0.04)), ((-10, -8, -15), (0, 0, 0.0)),
            ((0, 0, 0), (0.5, 0, 0)), ((0, 0, 45), (0.6, 0.3, 0)), ((12, -10, 45), (0.6, 0.3, -0.05)),
            ((0, 0, 90), (0.3, 0.8, 0)), ((-12, 10, 90), (0.3, 0.8, -0.05)), ((0, 0, -30), (-0.2, -0.3, -0.1))]
    return [tf.make_T(E(*r), np.array(t) + [0, 0, base_z]) for r, t in spec]


def generate(out_path: str, robot: str = "mock", hold: float = 3.0, move: float = 1.5, odom_hz: float = 20.0,
             lidar_hz: float = 10.0, pts_per_scan: int = 12_000, noise: float = 0.01, seed: int = 0):
    from builtin_interfaces.msg import Time
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header

    from .bag import write_bag

    rng = np.random.default_rng(seed)
    room = room_points(seed=seed)
    poses = station_poses()
    # Timeline: hold, move, hold, move, ...
    keys_t, keys_T = [], []
    t = 0.0
    for P in poses:
        keys_t += [t, t + hold]
        keys_T += [P, P]
        t += hold + move
    keys_t, T_end = np.array(keys_t), keys_t[-1]
    rots = Rotation.from_matrix(np.stack([T[:3, :3] for T in keys_T]))
    slerp = Slerp(keys_t, rots)
    trans = np.stack([T[:3, 3] for T in keys_T])

    def pose(tt):
        tt = np.clip(tt, keys_t[0], keys_t[-1])
        return tf.make_T(slerp([tt]).as_matrix()[0], [np.interp(tt, keys_t, trans[:, k]) for k in range(3)])

    t0_ns = 1_700_000_000 * 10 ** 9

    def stamp(tt):
        ns = t0_ns + int(tt * 1e9)
        return Time(sec=ns // 10 ** 9, nanosec=ns % 10 ** 9), ns

    odom_msgs, lidar_msgs = [], []
    dt = 1.0 / odom_hz
    for tt in np.arange(0, T_end, dt):
        T, T2 = pose(tt), pose(tt + dt)
        st, ns = stamp(tt)
        m = Odometry()
        m.header.stamp, m.header.frame_id, m.child_frame_id = st, f"{robot}/odom", f"{robot}/body"
        p = T[:3, 3]
        q = tf.quat_wxyz(T[:3, :3])
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = p
        m.pose.pose.orientation.w, m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z = q
        v = (T2[:3, 3] - p) / dt
        w = tf.so3_log(T[:3, :3].T @ T2[:3, :3]) / dt
        m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z = v + rng.normal(0, 0.002, 3)
        m.twist.twist.angular.x, m.twist.twist.angular.y, m.twist.twist.angular.z = w + rng.normal(0, 0.002, 3)
        odom_msgs.append((ns, m))
    for tt in np.arange(0, T_end, 1.0 / lidar_hz):
        L = pose(tt) @ X_TRUE  # odom_T_lidar
        P = tf.transform_points(tf.inv(L), room[rng.choice(len(room), pts_per_scan, replace=False)])
        P = P + rng.normal(0, noise, P.shape)
        st, ns = stamp(tt)
        msg: PointCloud2 = point_cloud2.create_cloud_xyz32(Header(stamp=st, frame_id=f"{robot}/livox"), P.astype(np.float32))
        lidar_msgs.append((ns + 5_000_000, msg))
    write_bag(out_path, odom_msgs, lidar_msgs, f"/{robot}/odom", f"/{robot}/livox/lidar")
    return X_TRUE
