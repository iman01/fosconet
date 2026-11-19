"""Prediction files and point cloud export."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import grid as G
from .bag import transform

KEEP_PROB = 0.05      # cells below this probability are not stored


def save_prediction(path, sample, grid: str, completion, model: str):
    """One scan's prediction: the scan cells, the hidden mask and every cell with probability >= KEEP_PROB."""
    idx = np.argwhere(completion.prob >= KEEP_PROB).astype(np.int16)
    coords, labels = sample.cells[grid]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, model=model, grid=grid, threshold=completion.threshold, stamp=sample.stamp, index=sample.index,
        T_map_ego=sample.T_map_ego, scan_coords=coords, scan_labels=labels,
        hidden=np.packbits(sample.vis[grid]['hidden']), coords=idx,
        prob=completion.prob[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float16), ground=completion.ground,
        seconds=completion.seconds)


class Prediction:
    """A saved prediction (see :func:`save_prediction`)."""

    def __init__(self, path):
        z = np.load(path)
        self.path = Path(path)
        self.model, self.grid = str(z['model']), str(z['grid'])
        self.threshold, self.stamp, self.index = float(z['threshold']), float(z['stamp']), int(z['index'])
        self.T_map_ego = z['T_map_ego']
        self.scan_coords, self.scan_labels = z['scan_coords'], z['scan_labels']
        n = int(np.prod(G.SHAPE[self.grid]))
        self.hidden = np.unpackbits(z['hidden'])[:n].astype(bool).reshape(G.SHAPE[self.grid])
        self.coords, self.prob, self.ground = z['coords'], z['prob'].astype(np.float32), z['ground']

    def occupied(self) -> np.ndarray:
        """Cells at or above the decision threshold (M, 3)."""
        return self.coords[self.prob >= self.threshold]

    def completed(self) -> np.ndarray:
        """Predicted occupied cells hidden from the scan inside the 2-6 m annulus, where the models were trained and
        evaluated: the completion of the occluded space (M, 3)."""
        c = self.occupied()
        c = c[G.annulus(self.grid)[c[:, 0], c[:, 1]]]
        return c[self.hidden[c[:, 0], c[:, 1], c[:, 2]]]

    def layer(self, coords: np.ndarray, lo: float = 0.1, hi: float = 3.0) -> np.ndarray:
        """Mask of cells between lo and hi metres above the estimated terrain (the layer a ground robot meets)."""
        h = (coords[:, 2] - self.ground[coords[:, 0], coords[:, 1]]) * G.DZ
        return (h >= lo) & (h <= hi)


COLOURS = {'ground': (150, 110, 70), 'stem': (200, 60, 30), 'completed': (20, 140, 190)}


def write_ply(path, xyz: np.ndarray, rgb: np.ndarray):
    """Binary little-endian PLY with x y z (float32) and red green blue (uint8)."""
    xyz = np.asarray(xyz, np.float32)
    rgb = np.asarray(rgb, np.uint8)
    head = ('ply\nformat binary_little_endian 1.0\n'
            f'element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode()
    rec = np.zeros(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    rec['x'], rec['y'], rec['z'] = xyz.T
    rec['r'], rec['g'], rec['b'] = rgb.T
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(head)
        f.write(rec.tobytes())


def export_ply(pred: Prediction, path, frame: str = 'ego'):
    """Scan cells (ground brown, stems red) and completed hidden cells (blue) as cell centres."""
    parts = [(pred.scan_coords[pred.scan_labels == G.GROUND], COLOURS['ground']),
             (pred.scan_coords[pred.scan_labels == G.STEM], COLOURS['stem']),
             (pred.completed(), COLOURS['completed'])]
    xyz = np.concatenate([G.centres(c, pred.grid) for c, _ in parts])
    rgb = np.concatenate([np.tile(col, (len(c), 1)) for c, col in parts])
    if frame == 'map':
        xyz = transform(pred.T_map_ego, xyz.astype(np.float64))
    write_ply(path, xyz, rgb)
