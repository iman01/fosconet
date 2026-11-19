"""Network inputs of one scan: sparse cell features and dense decoder channels."""
from __future__ import annotations

import numpy as np

from . import grid as G
from .model import Config

HEIGHT_SCALE = G.DZ / 3.0      # height above the terrain in units of 3 m, clipped to [-1, 2]


def ground_plane(coords: np.ndarray, labels: np.ndarray, grid: str) -> np.ndarray:
    """Terrain height (z cell index) of every column from a robust plane fitted to the scan's ground cells.

    The plane is refitted four times, each time keeping the cells within max(3 cells, 2.5 robust sigma) of it, so
    that low shrubs and stumps labelled as ground do not lift it. With fewer than 30 ground cells a horizontal
    plane at their median height is used.
    """
    X, Y = G.column_xy(grid)
    gc = np.asarray(coords)[np.asarray(labels) == G.GROUND]
    if len(gc) < 30:
        if len(gc):
            z0 = np.median(gc[:, 2])
        elif len(coords):
            z0 = np.percentile(np.asarray(coords)[:, 2], 5)
        else:
            z0 = (-1.5 - G.Z_MIN) / G.DZ
        return np.full(X.shape, int(round(z0)), np.int64)
    x, y, z = X[gc[:, 0], gc[:, 1]], Y[gc[:, 0], gc[:, 1]], gc[:, 2].astype(float)
    keep = np.ones(len(z), bool)
    for _ in range(4):
        A = np.stack([x[keep], y[keep], np.ones(keep.sum())], 1)
        c, *_ = np.linalg.lstsq(A, z[keep], rcond=None)
        res = z - (c[0] * x + c[1] * y + c[2])
        keep = np.abs(res) < max(3.0, 2.5 * 1.4826 * np.median(np.abs(res[keep])))
    return np.round(c[0] * X + c[1] * Y + c[2]).astype(np.int64)


def build(coords: np.ndarray, labels: np.ndarray, cfg: Config, vis: dict | None = None) -> dict:
    """Features and dense channels of one scan.

    coords: (M, 3) scan cells, labels: (M,) ground / stem class of each cell, vis: output of
    :func:`fosconet.visibility.trace` (needed when ``cfg.visibility``).
    Returns ``coords`` (M, 3) int32, ``feats`` (M, C) float32, ``dense`` (D, *grid) float16 or None and ``ground``
    (the terrain height per column, z index).
    """
    coords = np.asarray(coords).astype(np.int64)
    labels = np.asarray(labels).astype(np.uint8)
    g = ground_plane(coords, labels, cfg.grid)
    dense = []
    if cfg.visibility:
        if vis is None:
            raise ValueError('this model needs the visibility of the scan (fosconet.visibility.trace)')
        dense += [vis['seen'], vis['hidden']]
    if cfg.height:
        hag = (np.arange(G.NZ)[None, None, :] - g[:, :, None]) * HEIGHT_SCALE
        dense.append(np.clip(hag, -1.0, 2.0))
    feats = np.zeros((len(coords), cfg.in_channels), np.float32)
    feats[:, 0] = 1.0
    if cfg.labels:
        feats[:, 1] = labels == G.GROUND
        feats[:, 2] = labels == G.STEM
    if cfg.height:
        feats[:, 3] = np.clip((coords[:, 2] - g[coords[:, 0], coords[:, 1]]) * HEIGHT_SCALE, -1.0, 2.0)
    return dict(coords=coords.astype(np.int32), feats=feats,
                dense=np.stack(dense).astype(np.float16) if dense else None, ground=g)
