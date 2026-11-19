"""From a recording to network inputs: ego-frame scans, ground / stem labels, grids and visibility."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import grid as G
from . import visibility as V
from .bag import Recording
from .segment import REFLECTIVITY_GAIN, PastVote, Segmenter


@dataclass
class Sample:
    """One scan prepared for FOSCo-Net.

    xyz: points in the ego frame (N, 3), labels: their class (N,), T_map_ego: pose of the ego frame in the map,
    cells: {grid: (coords (M, 3), labels (M,))} ground and stem cells, vis: {grid: visibility volumes}.
    """
    index: int
    stamp: float
    T_map_ego: np.ndarray
    xyz: np.ndarray
    labels: np.ndarray
    cells: dict = field(default_factory=dict)
    vis: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f'{self.index:06d}_{self.stamp:.3f}'

    def save(self, path, keep_points: bool = True):
        d = dict(index=self.index, stamp=self.stamp, T_map_ego=self.T_map_ego, grids=np.array(list(self.cells)))
        if keep_points:
            d.update(xyz=self.xyz.astype(np.float32), labels=self.labels.astype(np.uint8))
        for g, (c, lb) in self.cells.items():
            d[f'{g}_coords'], d[f'{g}_labels'] = c, lb
            d.update({f'{g}_{k}': v for k, v in V.pack(self.vis[g]).items()})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path) -> 'Sample':
        z = np.load(path)
        s = cls(int(z['index']), float(z['stamp']), z['T_map_ego'], z['xyz'] if 'xyz' in z else np.zeros((0, 3)),
                z['labels'] if 'labels' in z else np.zeros(0, np.uint8))
        for g in z['grids']:
            g = str(g)
            s.cells[g] = (z[f'{g}_coords'], z[f'{g}_labels'])
            s.vis[g] = V.unpack({k[len(g) + 1:]: z[k] for k in z.files if k.startswith(f'{g}_vis_')}, g)
        return s


def prepare(recording, segmenter: Segmenter, grids=('polar',), every: int = 20, start: float | None = None,
            end: float | None = None, deskew: bool = True, log=print):
    """Yield a :class:`Sample` for every n-th scan of a recording (between start and end, seconds from its first
    scan). Every scan in the range is segmented, because the labels of a scan are voted over the past 5 s."""
    rec = recording if isinstance(recording, Recording) else Recording(recording)
    vote = PastVote(segmenter)
    pending = []
    t0 = None if start is None else max(0.0, start - 6.0)
    n = 0

    def emit(sc, xyz_map, ego, keep, T):
        lab = vote.labels(sc.index, xyz_map)
        s = Sample(sc.index, sc.stamp, T, ego[keep], lab[keep])
        for g in grids:
            s.cells[g] = G.voxelize(s.xyz, s.labels, g)
            s.vis[g] = V.trace(s.xyz, g)
        return s

    for sc in rec.scans(start=t0, end=end):
        xyz_map = rec.map_points(sc, deskew)
        vote.add(sc.index, xyz_map, REFLECTIVITY_GAIN * sc.reflectivity)
        t_rel = sc.t - rec.pose_t[0]
        if sc.index % every == 0 and (start is None or t_rel >= start):
            ego, keep, T = rec.ego_points(sc, deskew)
            pending.append((sc, xyz_map, ego, keep, T))
        while pending and vote.ready(pending[0][0].index):
            yield emit(*pending.pop(0))
            n += 1
    vote.finish()
    for p in pending:
        yield emit(*p)
        n += 1
    log(f'{n} samples')
