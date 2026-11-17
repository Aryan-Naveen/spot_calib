"""ROS 2 bag I/O (rosbag2_py; requires a sourced ROS 2 install). Bag receive time is used as the
common clock, so Livox/Spot header clocks need not be synchronised (stations are static)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..common import transforms as tf

log = logging.getLogger(__name__)


def _reader(path: str, topics: list[str]):
    import rosbag2_py

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id=""),
           rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    missing = [t for t in topics if t not in types]
    if missing:
        raise RuntimeError(f"topics not in bag: {missing}. Available: {sorted(types)}")
    r.set_filter(rosbag2_py.StorageFilter(topics=topics))
    return r, types


@dataclass
class Odom:
    t: np.ndarray  # (N,) bag time [s]
    T: np.ndarray  # (N,4,4) odom_T_body
    lin: np.ndarray  # (N,) |v| [m/s]
    ang: np.ndarray  # (N,) |w| [rad/s]
    frame_id: str
    child_frame_id: str


def read_odom(path: str, topic: str) -> Odom:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    r, types = _reader(path, [topic])
    cls = get_message(types[topic])
    ts, Ts, lin, ang = [], [], [], []
    frame = child = ""
    while r.has_next():
        _, data, t = r.read_next()
        m = deserialize_message(data, cls)
        p, q = m.pose.pose.position, m.pose.pose.orientation
        Ts.append(tf.make_T(tf.R_from_quat_wxyz(q.w, q.x, q.y, q.z), [p.x, p.y, p.z]))
        v, w = m.twist.twist.linear, m.twist.twist.angular
        lin.append(np.linalg.norm([v.x, v.y, v.z]))
        ang.append(np.linalg.norm([w.x, w.y, w.z]))
        ts.append(t * 1e-9)
        frame, child = m.header.frame_id, m.child_frame_id
    if not ts:
        raise RuntimeError(f"no messages on {topic}")
    o = Odom(np.array(ts), np.stack(Ts), np.array(lin), np.array(ang), frame, child)
    # Fall back to finite-difference speeds if the twist is not populated.
    if np.all(o.lin == 0) and np.all(o.ang == 0) and len(ts) > 2:
        dt = np.maximum(np.diff(o.t), 1e-3)
        dp = np.linalg.norm(np.diff(o.T[:, :3, 3], axis=0), axis=1) / dt
        dr = np.array([np.linalg.norm(tf.so3_log(a[:3, :3].T @ b[:3, :3])) for a, b in zip(o.T[:-1], o.T[1:])]) / dt
        o.lin = np.concatenate([[dp[0]], dp])
        o.ang = np.concatenate([[dr[0]], dr])
    log.info("odom: %d msgs over %.1fs (%s -> %s)", len(ts), o.t[-1] - o.t[0], frame, child)
    return o


def pointcloud2_xyz(msg) -> np.ndarray:
    offs = {f.name: f.offset for f in msg.fields}
    if not {"x", "y", "z"} <= offs.keys():
        raise RuntimeError("PointCloud2 has no x/y/z fields")
    dt = np.dtype({"names": ["x", "y", "z"], "formats": [">f4" if msg.is_bigendian else "<f4"] * 3,
                   "offsets": [offs["x"], offs["y"], offs["z"]], "itemsize": msg.point_step})
    n = msg.width * msg.height
    a = np.frombuffer(bytes(msg.data), dtype=dt, count=n)
    P = np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)
    ok = np.all(np.isfinite(P), axis=1) & (np.abs(P).sum(axis=1) > 1e-6)
    return P[ok]


def read_lidar_windows(path: str, topic: str, windows: list[tuple[float, float]], max_frames: int = 20) -> list[list[np.ndarray]]:
    """Collect up to ``max_frames`` clouds (lidar frame) for each (t0, t1) bag-time window."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    r, types = _reader(path, [topic])
    if types[topic] != "sensor_msgs/msg/PointCloud2":
        raise RuntimeError(f"{topic} is {types[topic]}; set livox xfer_format: 0 (PointCloud2)")
    cls = get_message(types[topic])
    out: list[list[np.ndarray]] = [[] for _ in windows]
    order = np.argsort([w[0] for w in windows])
    k = 0
    n = 0
    while r.has_next() and k < len(order):
        _, data, t = r.read_next()
        t *= 1e-9
        while k < len(order) and t > windows[order[k]][1]:
            k += 1
        if k >= len(order):
            break
        i = order[k]
        if windows[i][0] <= t <= windows[i][1] and len(out[i]) < max_frames:
            out[i].append(pointcloud2_xyz(deserialize_message(data, cls)))
            n += 1
    log.info("lidar: read %d clouds into %d stations", n, len(windows))
    return out


def write_bag(path: str, odom_msgs: list, lidar_msgs: list, odom_topic: str, lidar_topic: str):
    """Write (t_ns, msg) lists to a new bag (used by the mock generator)."""
    import rosbag2_py
    from rclpy.serialization import serialize_message

    w = rosbag2_py.SequentialWriter()
    last = None
    for sid in ("mcap", "sqlite3"):
        try:
            w.open(rosbag2_py.StorageOptions(uri=path, storage_id=sid), rosbag2_py.ConverterOptions("cdr", "cdr"))
            break
        except Exception as e:  # plugin missing
            last = e
            w = rosbag2_py.SequentialWriter()
    else:
        raise RuntimeError(f"cannot open bag for writing: {last}")
    for i, (name, typ) in enumerate([(odom_topic, "nav_msgs/msg/Odometry"), (lidar_topic, "sensor_msgs/msg/PointCloud2")]):
        try:
            meta = rosbag2_py.TopicMetadata(id=i, name=name, type=typ, serialization_format="cdr")
        except TypeError:
            meta = rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format="cdr")
        w.create_topic(meta)
    msgs = [(t, odom_topic, m) for t, m in odom_msgs] + [(t, lidar_topic, m) for t, m in lidar_msgs]
    msgs.sort(key=lambda x: x[0])
    for t, topic, m in msgs:
        w.write(topic, serialize_message(m), int(t))
    del w
