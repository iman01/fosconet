"""Reading the forest walks recordings (ROS 2 MCAP) without a ROS installation.

The recordings carry the raw Ouster scans (``/ouster/points``, frame ``os_lidar``), the optimized trajectory of the
rig (``/tf``: ``map`` -> ``body`` for every scan) and the static transforms of the rig (``/tf_static``). ``map`` is
a local East-North-Up frame, so its z axis is aligned with gravity.

The ego frame of a scan, in which the grids are built, has its origin at the LiDAR, z along gravity from the Xsens
AHRS (median over +-2 s) and x along the rig's backward heading (the body x axis, levelled and rotated by 180 deg),
as in the training data. Points are deskewed to the end of the sweep, the time of the scan's trajectory pose.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCAN_TOPIC = '/ouster/points'
TF_TOPIC = '/tf'
TF_STATIC_TOPIC = '/tf_static'
AHRS_TOPIC = '/imu/data'   # Xsens MTi-30 orientation (ENU): the gravity direction that levels the ego frame
UP_WINDOW = 2.0            # s: median of the gravity direction over +-this window
LIDAR_FRAME = 'os_lidar'
OPERATOR_RADIUS = 1.2      # m: points closer to the LiDAR ...
OPERATOR_BELOW = -1.0      # ... and higher than this relative to it are the operator or the rig
POINT_STRIDE = 3           # every 3rd point of the raw scan, as the odometry that produced the training scans kept


# ----------------------------------------------------------------------------------------------- rotations
def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(..., 4) quaternions x, y, z, w -> (..., 3, 3) rotation matrices."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def slerp(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    q0, q1, u = np.asarray(q0, float), np.asarray(q1, float).copy(), np.asarray(u, float)[..., None]
    d = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.abs(d)
    th = np.arccos(np.clip(d, -1, 1))
    s = np.sin(th)
    small = s < 1e-6
    w0 = np.where(small, 1 - u, np.sin((1 - u) * th) / np.where(small, 1, s))
    w1 = np.where(small, u, np.sin(u * th) / np.where(small, 1, s))
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def transform(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


# ----------------------------------------------------------------------------------------------- data
@dataclass
class Scan:
    """One LiDAR scan with its pose.

    xyz: points in the os_lidar frame (N, 3); dt: time of each point after the scan start [s];
    reflectivity: Ouster calibrated reflectivity (N,); stamp: scan start [s]; t: time of the scan's trajectory
    pose [s]; index: position in the recording.
    """
    index: int
    stamp: float
    t: float
    xyz: np.ndarray
    dt: np.ndarray
    reflectivity: np.ndarray


def _reader(path):
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as e:  # pragma: no cover
        raise ImportError('reading the recordings needs rosbags: pip install rosbags') from e
    return AnyReader([Path(path)], default_typestore=get_typestore(Stores.ROS2_HUMBLE))


def _cloud(msg, stride: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    off = {f.name: f.offset for f in msg.fields}
    buf = np.frombuffer(bytes(msg.data), np.uint8).reshape(-1, msg.point_step)[::stride]

    def col(name, dtype, size):
        return buf[:, off[name]:off[name] + size].copy().view(dtype).ravel()
    xyz = np.stack([col(k, np.float32, 4) for k in 'xyz'], 1)
    dt = col('t', np.uint32, 4).astype(np.float64) * 1e-9
    refl = col('reflectivity', np.uint16, 2).astype(np.float32)
    ok = np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 0)
    return xyz[ok], dt[ok], refl[ok]


class Recording:
    """A walks recording (a bag folder with the .mcap file and metadata.yaml).

    point_stride: keep every n-th point of the raw scan order. The training scans of the walks dataset hold every
    3rd point (the setting of the LiDAR odometry during recording), so 3 reproduces their density.

    >>> rec = Recording('walks_2026-08-12_sample60s')
    >>> for scan in rec.scans(every=20):
    ...     xyz_ego, T_map_ego = rec.ego_points(scan)
    """

    def __init__(self, path, point_stride: int = POINT_STRIDE):
        self.point_stride = int(point_stride)
        self.path = Path(path)
        if self.path.suffix == '.mcap':
            self.path = self.path.parent
        t, q, p = [], [], []
        at, aq = [], []
        static = {}
        with _reader(self.path) as r:
            names = {c.topic for c in r.connections}
            for topic in (SCAN_TOPIC, TF_TOPIC, TF_STATIC_TOPIC):
                if topic not in names:
                    raise ValueError(f'{self.path}: topic {topic} missing (is this a walks recording?)')
            conns = [c for c in r.connections if c.topic in (TF_TOPIC, TF_STATIC_TOPIC, AHRS_TOPIC)]
            for c, _, raw in r.messages(connections=conns):
                m = r.deserialize(raw, c.msgtype)
                if c.topic == AHRS_TOPIC:
                    o = m.orientation
                    at.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
                    aq.append([o.x, o.y, o.z, o.w])
                    continue
                for tr in m.transforms:
                    a, b = tr.transform.translation, tr.transform.rotation
                    if c.topic == TF_STATIC_TOPIC:
                        static[(tr.header.frame_id, tr.child_frame_id)] = make_T(quat_to_mat([b.x, b.y, b.z, b.w]),
                                                                                 [a.x, a.y, a.z])
                    elif tr.header.frame_id == 'map' and tr.child_frame_id == 'body':
                        t.append(tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9)
                        q.append([b.x, b.y, b.z, b.w])
                        p.append([a.x, a.y, a.z])
            self.n_scans = next(c.msgcount for c in r.connections if c.topic == SCAN_TOPIC)
        o = np.argsort(t)
        self.pose_t, self.pose_q, self.pose_p = np.asarray(t)[o], np.asarray(q)[o], np.asarray(p)[o]
        self.T_body_lidar = self._chain(static, 'body', LIDAR_FRAME)
        self._up = self._ahrs_up(np.asarray(at), np.asarray(aq)) if len(at) > 10 else None

    def _ahrs_up(self, at, aq) -> np.ndarray:
        """Gravity up (map frame) at every trajectory pose from the nearest AHRS orientation."""
        o = np.argsort(at)
        at, aq = at[o], aq[o]
        j = np.clip(np.searchsorted(at, self.pose_t), 1, len(at) - 1)
        j = np.where(np.abs(at[j - 1] - self.pose_t) < np.abs(at[j] - self.pose_t), j - 1, j)
        up_imu = quat_to_mat(aq[j])[:, 2, :]                  # R_world_imu^T e_z
        up = np.einsum('nij,nj->ni', quat_to_mat(self.pose_q), up_imu)
        up[np.abs(at[j] - self.pose_t) > 0.02] = np.nan
        return up

    def up(self, t: float) -> np.ndarray:
        """Gravity up in the map frame at time t: median of the AHRS direction over +-UP_WINDOW (map z without an
        AHRS)."""
        if self._up is None:
            return np.array([0.0, 0.0, 1.0])
        lo, hi = np.searchsorted(self.pose_t, [t - UP_WINDOW, t + UP_WINDOW + 1e-9])
        u = self._up[lo:hi]
        u = u[np.isfinite(u).all(1)]
        u = np.median(u, axis=0) if len(u) else np.array([0.0, 0.0, 1.0])
        return u / np.linalg.norm(u)

    @staticmethod
    def _chain(static, parent, child) -> np.ndarray:
        """Static transform parent -> child through the tree of /tf_static."""
        up = {c: (p, T) for (p, c), T in static.items()}
        T = np.eye(4)
        node = child
        while node != parent:
            if node not in up:
                raise ValueError(f'no static transform chain from {parent} to {child}')
            p, Tp = up[node]
            T = Tp @ T
            node = p
        return T

    # ------------------------------------------------------------------------------------------- poses
    def body_pose(self, t) -> np.ndarray:
        """T_map_body at the times t (scalar or array), interpolated (slerp) between trajectory poses."""
        t = np.atleast_1d(np.asarray(t, np.float64))
        i = np.clip(np.searchsorted(self.pose_t, t), 1, len(self.pose_t) - 1)
        t0, t1 = self.pose_t[i - 1], self.pose_t[i]
        u = np.clip((t - t0) / np.maximum(t1 - t0, 1e-9), 0, 1)
        q = slerp(self.pose_q[i - 1], self.pose_q[i], u)
        p = self.pose_p[i - 1] + u[:, None] * (self.pose_p[i] - self.pose_p[i - 1])
        T = np.tile(np.eye(4), (len(t), 1, 1))
        T[:, :3, :3], T[:, :3, 3] = quat_to_mat(q), p
        return T

    def lidar_pose(self, t) -> np.ndarray:
        return self.body_pose(t) @ self.T_body_lidar

    def ego_frame(self, t: float) -> np.ndarray:
        """T_map_ego of a scan whose pose time is t."""
        Tb = self.body_pose(t)[0]
        origin = (Tb @ self.T_body_lidar)[:3, 3]
        z = self.up(t)
        h = Tb[:3, 0] - (Tb[:3, 0] @ z) * z
        x = -h / np.linalg.norm(h)
        return make_T(np.stack([x, np.cross(z, x), z], 1), origin)

    # ------------------------------------------------------------------------------------------- scans
    def scans(self, start: float | None = None, end: float | None = None, every: int = 1):
        """Scans in time order (every n-th), between start and end (seconds from the first scan, optional)."""
        with _reader(self.path) as r:
            conns = [c for c in r.connections if c.topic == SCAN_TOPIC]
            t_first, idx = None, -1
            for c, tb, raw in r.messages(connections=conns):
                idx += 1
                tb = tb * 1e-9
                t_first = tb if t_first is None else t_first
                rel = tb - t_first
                if start is not None and rel < start:
                    continue
                if end is not None and rel > end:
                    break
                if idx % every:
                    continue
                yield self.make_scan(idx, tb, r.deserialize(raw, c.msgtype))

    def make_scan(self, index: int, t_bag: float, msg) -> Scan:
        xyz, dt, refl = _cloud(msg, self.point_stride)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        j = int(np.argmin(np.abs(self.pose_t - t_bag)))       # the trajectory pose of this scan
        return Scan(index, stamp, float(self.pose_t[j]), xyz, dt, refl)

    def map_points(self, scan: Scan, deskew: bool = True) -> np.ndarray:
        """Points of a scan in the map frame. With deskew, each point is moved with the LiDAR pose at its own time,
        counted back from the scan's pose, which belongs to the end of the sweep."""
        if not deskew:
            return transform(self.lidar_pose(scan.t)[0], scan.xyz.astype(np.float64))
        tu, inv = np.unique(np.round(scan.t - (scan.dt.max() - scan.dt), 6), return_inverse=True)
        T = self.lidar_pose(tu)
        p = scan.xyz.astype(np.float64)
        return np.einsum('nij,nj->ni', T[inv, :3, :3], p) + T[inv, :3, 3]

    def ego_points(self, scan: Scan, deskew: bool = True, remove_operator: bool = True):
        """Points of a scan in its ego frame, the mask of kept points and T_map_ego."""
        T = self.ego_frame(scan.t)
        p = transform(np.linalg.inv(T), self.map_points(scan, deskew))
        keep = np.ones(len(p), bool)
        if remove_operator:
            keep = ~((np.linalg.norm(p, axis=1) < OPERATOR_RADIUS) & (p[:, 2] > OPERATOR_BELOW))
        return p.astype(np.float32), keep, T
