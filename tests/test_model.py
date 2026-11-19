import numpy as np
import pytest
import torch

from fosconet import grid as G
from fosconet.predict import MODELS, Predictor

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='spconv inference needs a GPU')


def synthetic_scan(grid):
    """Flat ground 1.5 m below the sensor and one trunk 3 m away."""
    xs = np.linspace(-6, 6, 241)
    X, Y = np.meshgrid(xs, xs)
    ground = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, -1.55)])
    zz = np.linspace(-1.5, 3.0, 46)
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    trunk = np.column_stack([3.0 + 0.2 * np.cos(np.repeat(t, len(zz))), 0.2 * np.sin(np.repeat(t, len(zz))),
                             np.tile(zz, len(t))])
    xyz = np.concatenate([ground, trunk]).astype(np.float32)
    lab = np.r_[np.full(len(ground), G.GROUND), np.full(len(trunk), G.STEM)]
    return xyz, lab


@cuda
@pytest.mark.parametrize('name', sorted(MODELS))
def test_published_models_run(name):
    from fosconet import visibility as V
    p = Predictor(name)
    xyz, lab = synthetic_scan(p.grid)
    c, lb = G.voxelize(xyz, lab, p.grid)
    out = p(c, lb, V.trace(xyz, p.grid))
    assert out.prob.shape == G.SHAPE[p.grid]
    assert 0.0 <= float(out.prob.min()) and float(out.prob.max()) <= 1.0
    assert 0.0 < out.threshold < 1.0


@cuda
def test_polar_seam_equivariance():
    """FOSCo-Net 2.0 (polar) is equivariant to rotations by multiples of 8 deg."""
    from fosconet import visibility as V
    p = Predictor('fosconet2-polar', half=False)
    xyz, lab = synthetic_scan('polar')
    c, lb = G.voxelize(xyz, lab, 'polar')
    v = V.trace(xyz, 'polar')
    a = p(c, lb, v).prob.astype(np.float32)
    c2 = c.copy()
    c2[:, 1] = (c2[:, 1] + 16) % G.NT
    v2 = {k: np.roll(x, 16, axis=1) for k, x in v.items()}
    b = p(c2, lb, v2).prob.astype(np.float32)
    assert np.abs(np.roll(a, 16, axis=1) - b).max() < 1e-2
