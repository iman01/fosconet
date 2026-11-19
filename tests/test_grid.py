import numpy as np

from fosconet import grid as G
from fosconet import visibility as V


def test_point_cells_and_centres_round_trip():
    rng = np.random.default_rng(0)
    for g in G.GRIDS:
        c = np.column_stack([rng.integers(0, n, 500) for n in G.SHAPE[g]])
        if g == 'cart':
            c = c[G.annulus(g)[c[:, 0], c[:, 1]]]
        back, ok = G.point_cells(G.centres(c, g), g)
        assert ok.all()
        assert (back == c).all()


def test_voxelize_keeps_ground_and_stem_with_stem_majority():
    xyz = np.array([[3.0, 0.05, -1.45]] * 3 + [[3.0, 0.05, 0.55]] * 2 + [[4.0, 1.0, 0.0]], np.float32)
    labels = np.array([G.GROUND, G.GROUND, G.STEM, G.STEM, G.STEM, G.CANOPY])
    c, lb = G.voxelize(xyz, labels, 'polar')
    assert len(c) == 2                                   # the canopy point is dropped
    assert sorted(lb.tolist()) == [G.GROUND, G.STEM]


def test_visibility_wall():
    y, z = np.meshgrid(np.linspace(-1, 1, 81), np.linspace(-1, 1, 81))
    wall = np.column_stack([np.full(y.size, 4.0), y.ravel(), z.ravel()])
    v = V.trace(wall, 'polar')
    a = lambda r: int((r - G.R_MIN) / G.DR)          # noqa: E731
    k = int((0.0 - G.Z_MIN) / G.DZ)
    assert v['seen'][a(3.0), 0, k] and v['hidden'][a(5.0), 0, k] and v['hit'][a(4.0), 0, k]
    assert v['hidden'][a(3.0), 180, k]                  # no return behind the sensor: hidden
    assert not (v['seen'] & v['hidden']).any()
