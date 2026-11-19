"""The joint map of the dataset as a visual reference: the map cells around a scan, in its ego frame.

The map (``map/walks_2026-08-12_map_local_enu.pcd`` in the companion archive) accumulates all walks of the day in
the ``map`` frame of the recordings. Cropped around a scan it shows what the occluded space actually holds. It is
unlabelled and keeps every class, so it is a qualitative reference, not the benchmark's ground truth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import grid as G
from .bag import transform


def read_pcd(path) -> np.ndarray:
    """x, y, z of a PCD file (binary or ascii, float32 fields), as a memory-mapped (N, F) array when binary."""
    path = Path(path)
    with open(path, 'rb') as f:
        head = {}
        while True:
            line = f.readline().decode('ascii', 'replace').strip()
            k, *v = line.split()
            head[k] = v
            if k == 'DATA':
                break
        offset = f.tell()
    fields, n = head['FIELDS'], int(head['POINTS'][0])
    if head['DATA'][0] == 'binary':
        if set(head['SIZE']) != {'4'} or set(head['TYPE']) != {'F'}:
            raise ValueError('only float32 PCD fields are supported')
        a = np.memmap(path, np.float32, 'r', offset, (n, len(fields)))
    elif head['DATA'][0] == 'ascii':
        a = np.loadtxt(path, np.float32, skiprows=len(head))
    else:
        raise ValueError(f'unsupported PCD data mode {head["DATA"][0]}')
    return a[:, [fields.index(k) for k in 'xyz']] if fields[:3] != ['x', 'y', 'z'] else a[:, :3]


class MapReference:
    """Map cells around scans. Loads the part of the map near the given ego origins once."""

    def __init__(self, pcd, origins: np.ndarray | None = None, margin: float = 8.0):
        a = read_pcd(pcd)
        if origins is not None:
            o = np.asarray(origins)
            lo, hi = o.min(0) - margin, o.max(0) + margin
            m = np.ones(len(a), bool)
            for i in range(3):
                col = np.asarray(a[:, i])
                m &= (col >= lo[i] - (20 if i == 2 else 0)) & (col <= hi[i] + (20 if i == 2 else 0))
            a = np.asarray(a[m])
        self.xyz = np.asarray(a, np.float64)

    def cells(self, T_map_ego: np.ndarray, grid: str) -> np.ndarray:
        o = T_map_ego[:3, 3]
        near = np.abs(self.xyz[:, :2] - o[:2]).max(1) < 6.5
        p = transform(np.linalg.inv(T_map_ego), self.xyz[near])
        c, ok = G.point_cells(p.astype(np.float32), grid)
        return G.unlin(np.unique(G.lin(c[ok], grid)), grid)
