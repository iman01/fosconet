"""What a single scan saw: a depth-image test of every grid cell.

A spherical depth image with 1 x 1 deg pixels holds the range of the nearest return per direction. Pixels without
a return take the minimum of their 3 x 3 neighbourhood (azimuth wraps), which closes the gaps of sparse beam
patterns on the conservative side. Every cell centre is then compared with the depth of its pixel:

=========  =======================================================================
``seen``   closer than the surface by more than ``MARGIN``: observed empty
``margin`` within ``MARGIN`` of the surface: undecided
``hidden`` behind the surface by more than ``MARGIN``, or in a direction without a return
``hit``    the cell holds a return
=========  =======================================================================

The networks receive ``seen`` and ``hidden`` as dense input channels.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import minimum_filter

from . import grid as G

RES = 1.0                     # deg, azimuth and elevation
NAZ, NEL = int(360 / RES), int(180 / RES)
MARGIN = 0.15                 # m


def _pixels(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.linalg.norm(v, axis=1)
    az = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
    el = np.degrees(np.arcsin(np.clip(v[:, 2] / np.maximum(rng, 1e-9), -1, 1))) + 90.0
    pix = np.clip((el / RES).astype(np.int64), 0, NEL - 1) * NAZ + np.clip((az / RES).astype(np.int64), 0, NAZ - 1)
    return pix, rng


@lru_cache(maxsize=2)
def _geometry(grid: str) -> tuple[np.ndarray, np.ndarray]:
    pix, rng = _pixels(G.all_centres(grid).astype(np.float64))
    return pix.astype(np.int32), rng.astype(np.float32)


def depth_image(xyz: np.ndarray) -> np.ndarray:
    """Range of the nearest return per 1-deg pixel (NEL, NAZ); gaps filled from the 3 x 3 neighbourhood."""
    pix, d = _pixels(np.asarray(xyz, np.float64))
    D = np.full(NEL * NAZ, np.inf)
    np.minimum.at(D, pix, d)
    D = D.reshape(NEL, NAZ)
    filled = minimum_filter(D, size=3, mode=('nearest', 'wrap'))
    return np.where(np.isfinite(D), D, filled)


def trace(xyz: np.ndarray, grid: str) -> dict[str, np.ndarray]:
    """Visibility volumes of one scan (points in the ego frame, sensor at the origin).

    Returns dense bool volumes ``seen``, ``hit``, ``margin`` and ``hidden`` of the grid.
    """
    G.check_grid(grid)
    xyz = np.asarray(xyz, np.float64)
    xyz = xyz[np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 0.05)]
    pix, rng = _geometry(grid)
    dep = depth_image(xyz).ravel()[pix]
    fin = np.isfinite(dep)
    seen = (fin & (rng < dep - MARGIN)).reshape(G.SHAPE[grid])
    margin = (fin & (np.abs(rng - dep) <= MARGIN)).reshape(G.SHAPE[grid]) & ~seen
    hit = np.zeros(G.SHAPE[grid], bool)
    c, ok = G.point_cells(xyz.astype(np.float32), grid)
    c = c[ok]
    hit[c[:, 0], c[:, 1], c[:, 2]] = True
    seen &= ~hit
    margin &= ~hit
    return dict(seen=seen, hit=hit, margin=margin, hidden=~(seen | hit | margin))


def pack(vis: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Bit-packed copy for storage (``seen``, ``hit``, ``margin``)."""
    return {f'vis_{k}': np.packbits(vis[k]) for k in ('seen', 'hit', 'margin')}


def unpack(z, grid: str) -> dict[str, np.ndarray]:
    n = int(np.prod(G.SHAPE[grid]))
    out = {k: np.unpackbits(z[f'vis_{k}'])[:n].astype(bool).reshape(G.SHAPE[grid]) for k in ('seen', 'hit', 'margin')}
    out['hidden'] = ~(out['seen'] | out['hit'] | out['margin'])
    return out
