"""Ego-centric voxel grids of FOSCo-Net.

Both grids live in the gravity-levelled ego frame of a scan: origin at the LiDAR, z up, x along the rig's
backward heading (see :mod:`fosconet.bag`). Heights are relative to the sensor.

=======  ===================  ==============================================================================
grid     shape                cells
=======  ===================  ==============================================================================
polar    40 x 360 x 250       r 2.0-6.0 m (0.1 m), azimuth 0-360 deg (1 deg), z -5 to 20 m (0.1 m)
cart     120 x 120 x 250      x, y -6 to 6 m (0.1 m), z -5 to 20 m (0.1 m); scored inside the 2-6 m annulus
=======  ===================  ==============================================================================

Cells are addressed by integer indices ``(a, b, k)``: ``(r, theta, z)`` on the polar grid and ``(x, y, z)`` on the
Cartesian grid.
"""
from __future__ import annotations

import numpy as np

NR, NT, NZ = 40, 360, 250
R_MIN, R_MAX, DR = 2.0, 6.0, 0.1
THETA_DEG = 1.0
Z_MIN, Z_MAX, DZ = -5.0, 20.0, 0.1
NX, NY = 120, 120
X_MIN, X_MAX, DX = -6.0, 6.0, 0.1
Y_MIN, Y_MAX, DY = -6.0, 6.0, 0.1

SHAPE = {'polar': (NR, NT, NZ), 'cart': (NX, NY, NZ)}
GRIDS = tuple(SHAPE)

# semantic classes of the DigiForests segmenter; the grids keep ground and stem cells only
IGNORE, GROUND, SHRUB, STEM, CANOPY = 0, 1, 2, 3, 4
CLASS_NAMES = ('ignore', 'ground', 'shrub', 'stem', 'canopy')


def check_grid(grid: str) -> str:
    if grid not in SHAPE:
        raise ValueError(f"grid must be one of {GRIDS}, got {grid!r}")
    return grid


def point_cells(xyz: np.ndarray, grid: str) -> tuple[np.ndarray, np.ndarray]:
    """Cell index (N, 3) int64 of every point and a mask of the points inside the grid."""
    check_grid(grid)
    xyz = np.asarray(xyz, np.float32)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if grid == 'polar':
        r = np.sqrt(x ** 2 + y ** 2)
        ok = (r >= R_MIN) & (r < R_MAX) & (z >= Z_MIN) & (z < Z_MAX)
        a = np.clip(((r - R_MIN) / DR).astype(np.int64), 0, NR - 1)
        b = np.clip(((np.degrees(np.arctan2(y, x)) % 360.0) / THETA_DEG).astype(np.int64), 0, NT - 1)
    else:
        ok = (x >= X_MIN) & (x < X_MAX) & (y >= Y_MIN) & (y < Y_MAX) & (z >= Z_MIN) & (z < Z_MAX)
        a = np.clip(((x - X_MIN) / DX).astype(np.int64), 0, NX - 1)
        b = np.clip(((y - Y_MIN) / DY).astype(np.int64), 0, NY - 1)
    k = np.clip(((z - Z_MIN) / DZ).astype(np.int64), 0, NZ - 1)
    return np.column_stack([a, b, k]), ok


def lin(coords: np.ndarray, grid: str) -> np.ndarray:
    """Linear index of (M, 3) cell coordinates."""
    c = np.asarray(coords).astype(np.int64)
    s = SHAPE[grid]
    return (c[:, 0] * s[1] + c[:, 1]) * s[2] + c[:, 2]


def unlin(idx: np.ndarray, grid: str) -> np.ndarray:
    s = SHAPE[grid]
    idx = np.asarray(idx, np.int64)
    return np.column_stack([idx // (s[1] * s[2]), (idx // s[2]) % s[1], idx % s[2]]).astype(np.int16)


def voxelize(xyz: np.ndarray, labels: np.ndarray, grid: str, keep=(GROUND, STEM)) -> tuple[np.ndarray, np.ndarray]:
    """Occupied cells of a labelled scan.

    Only points of the classes in ``keep`` are used (the networks were trained on ground and stem cells). A cell is
    a stem cell when it holds strictly more stem than ground points, otherwise a ground cell.
    Returns the cell coordinates (M, 3) int16 and their labels (M,) uint8.
    """
    xyz = np.asarray(xyz, np.float32)
    labels = np.asarray(labels)
    m = np.isin(labels, keep)
    c, ok = point_cells(xyz[m], grid)
    c, lb = c[ok], labels[m][ok]
    if len(c) == 0:
        return np.zeros((0, 3), np.int16), np.zeros(0, np.uint8)
    u, inv = np.unique(lin(c, grid), return_inverse=True)
    ng = np.bincount(inv, weights=(lb == GROUND), minlength=len(u))
    ns = np.bincount(inv, weights=(lb == STEM), minlength=len(u))
    cls = np.where(ns > ng, STEM, GROUND).astype(np.uint8)
    return unlin(u, grid), cls


def centres(coords: np.ndarray, grid: str) -> np.ndarray:
    """Metric centre (M, 3) of every cell, in the ego frame."""
    c = np.asarray(coords).astype(np.float64)
    z = Z_MIN + (c[:, 2] + 0.5) * DZ
    if grid == 'polar':
        r = R_MIN + (c[:, 0] + 0.5) * DR
        t = np.radians((c[:, 1] + 0.5) * THETA_DEG)
        return np.column_stack([r * np.cos(t), r * np.sin(t), z]).astype(np.float32)
    return np.column_stack([X_MIN + (c[:, 0] + 0.5) * DX, Y_MIN + (c[:, 1] + 0.5) * DY, z]).astype(np.float32)


def all_centres(grid: str) -> np.ndarray:
    """Centres of all cells, in C order of the dense volume."""
    return centres(np.indices(SHAPE[grid]).reshape(3, -1).T, grid)


def column_xy(grid: str) -> tuple[np.ndarray, np.ndarray]:
    """Metric (x, y) of every column centre, shape SHAPE[grid][:2]."""
    if grid == 'polar':
        R = np.repeat((R_MIN + (np.arange(NR) + 0.5) * DR)[:, None], NT, axis=1)
        th = np.radians((np.arange(NT) + 0.5) * THETA_DEG)[None, :].repeat(NR, 0)
        return R * np.cos(th), R * np.sin(th)
    xc = X_MIN + (np.arange(NX) + 0.5) * DX
    yc = Y_MIN + (np.arange(NY) + 0.5) * DY
    return tuple(np.meshgrid(xc, yc, indexing='ij'))


def annulus(grid: str) -> np.ndarray:
    """Columns whose centre lies in the 2-6 m annulus (all columns of the polar grid)."""
    if grid == 'polar':
        return np.ones(SHAPE['polar'][:2], bool)
    X, Y = column_xy('cart')
    R = np.hypot(X, Y)
    return (R >= R_MIN) & (R < R_MAX)


def dense(coords: np.ndarray, grid: str, values=None, dtype=bool) -> np.ndarray:
    a = np.zeros(SHAPE[grid], dtype)
    if len(coords):
        c = np.asarray(coords).astype(np.int64)
        a[c[:, 0], c[:, 1], c[:, 2]] = True if values is None else values
    return a
