"""Load a published FOSCo-Net model and complete scans with it."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import grid as G
from . import inputs as I
from .model import Config, FOSCoNet

WEIGHTS_DIR = Path(__file__).resolve().parent / 'weights'

MODELS = {
    'fosconet2-polar': 'fosconet2_polar.pt',
    'fosconet2-cart': 'fosconet2_cart.pt',
    'fosconet1-polar': 'fosconet1_polar.pt',
    'fosconet1-cart': 'fosconet1_cart.pt',
}


def weights_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix == '.pt' and p.exists():
        return p
    name_or_path = name_or_path.replace('_', '-')
    if name_or_path in MODELS:
        f = WEIGHTS_DIR / MODELS[name_or_path]
        if f.exists():
            return f
        raise FileNotFoundError(f'{f} not found: the weights ship with the package in fosconet/weights/')
    raise ValueError(f'unknown model {name_or_path!r}; choose one of {sorted(MODELS)} or a .pt file')


@dataclass
class Completion:
    """Output of one scan. ``prob`` is the calibrated occupancy probability of every cell (dense, float16);
    ``occupied`` the cells at or above the model's decision threshold."""
    grid: str
    prob: np.ndarray
    threshold: float
    scan_coords: np.ndarray
    ground: np.ndarray
    seconds: float

    @property
    def occupied(self) -> np.ndarray:
        return self.prob >= self.threshold

    def completed_cells(self, hidden: np.ndarray | None = None) -> np.ndarray:
        """Cells predicted occupied that the scan does not hold (optionally only hidden ones), (M, 3) int16."""
        m = self.occupied & ~G.dense(self.scan_coords, self.grid)
        if hidden is not None:
            m &= hidden
        return np.argwhere(m).astype(np.int16)


class Predictor:
    """FOSCo-Net with its calibration and validation threshold.

    >>> p = Predictor('fosconet2-polar')
    >>> out = p(coords, labels, vis)        # scan cells, their ground / stem labels, visibility volumes
    >>> out.completed_cells(vis['hidden'])
    """

    def __init__(self, model: str = 'fosconet2-polar', device: str | None = None, half: bool = True):
        path = weights_path(model)
        ck = torch.load(path, map_location='cpu', weights_only=True)
        self.name = ck.get('name', path.stem)
        self.cfg = Config.from_dict(ck['config'])
        self.platt = ck.get('platt')
        self.threshold = float(ck['threshold'])
        self.meta = ck.get('meta', {})
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.half = half and self.device.type == 'cuda'
        self.net = FOSCoNet(self.cfg).to(self.device).eval()
        self.net.load_state_dict(ck['state_dict'])

    @property
    def grid(self) -> str:
        return self.cfg.grid

    @torch.no_grad()
    def logits(self, x: dict) -> torch.Tensor:
        coords = torch.from_numpy(np.concatenate([np.zeros((len(x['coords']), 1), np.int32), x['coords']], 1))
        feats = torch.from_numpy(x['feats'])
        dense = None if x['dense'] is None else torch.from_numpy(x['dense'])[None].to(self.device)
        if dense is not None and not self.half:
            dense = dense.float()
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.half):
            z = self.net(coords.to(self.device), feats.to(self.device), 1, dense)
        return z.float()[0, 0]

    def __call__(self, coords: np.ndarray, labels: np.ndarray, vis: dict | None = None) -> Completion:
        t0 = time.perf_counter()
        x = I.build(coords, labels, self.cfg, vis)
        z = self.logits(x)
        if self.platt is not None:
            z = self.platt[0] * z + self.platt[1]
        prob = torch.sigmoid(z).cpu().numpy().astype(np.float16)
        return Completion(self.grid, prob, self.threshold, np.asarray(coords).astype(np.int16), x['ground'],
                          time.perf_counter() - t0)
