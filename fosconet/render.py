"""Figures of predictions: 3D views and top views of the scan, the completion and the reference map."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import grid as G

C_GROUND = (0.62, 0.47, 0.33)
C_STEM = (0.80, 0.22, 0.12)
C_DONE = (0.10, 0.55, 0.78)
C_MAP = (0.45, 0.45, 0.45)
C_HIDDEN = (0.90, 0.91, 0.94)
VIEW = dict(elev=28, azim=-60)
NAMES = {'fosconet2-polar': 'FOSCo-Net 2.0 (polar)', 'fosconet2-cart': 'FOSCo-Net 2.0 (Cartesian)',
         'fosconet1-polar': 'FOSCo-Net 1.0 (polar)', 'fosconet1-cart': 'FOSCo-Net 1.0 (Cartesian)'}


def _height(coords, ground, grid):
    return (coords[:, 2] - ground[coords[:, 0], coords[:, 1]]) * G.DZ


def _layer(coords, ground, grid, lo=-0.3, hi=4.0):
    if len(coords) == 0:
        return coords
    h = _height(coords, ground, grid)
    return coords[(h >= lo) & (h <= hi)]


def _ground_on(grid_from, ground, grid_to):
    """Terrain height of one grid's columns sampled at another grid's column centres."""
    if grid_from == grid_to:
        return ground
    X, Y = G.column_xy(grid_to)
    c, ok = G.point_cells(np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)]).astype(np.float32), grid_from)
    g = ground[c[:, 0], c[:, 1]].reshape(X.shape)
    return np.where(ok.reshape(X.shape), g, int(np.median(ground)))


def _scatter3d(ax, layers, zlim):
    for xyz, col, s in layers:
        if len(xyz):
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=s, color=col, depthshade=False, lw=0)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((12, 12, zlim[1] - zlim[0]))
    ax.view_init(**VIEW)
    for f in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
        f([])
    ax.xaxis.pane.set_alpha(0.15)
    ax.yaxis.pane.set_alpha(0.15)
    ax.zaxis.pane.set_alpha(0.15)


def _top(ax, layers, hidden_xy=None):
    if hidden_xy is not None:
        ax.scatter(hidden_xy[:, 0], hidden_xy[:, 1], s=1.0, color=C_HIDDEN, lw=0, marker='s')
    for xyz, col, s in layers:
        if len(xyz):
            ax.scatter(xyz[:, 0], xyz[:, 1], s=s, color=col, lw=0)
    for r in (G.R_MIN, G.R_MAX):
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r * np.cos(t), r * np.sin(t), color=(0.6, 0.6, 0.6), lw=0.6)
    ax.set_xlim(-6.3, 6.3)
    ax.set_ylim(-6.3, 6.3)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def figure(preds: list, out, reference_cells=None, title: str | None = None, dpi: int = 200):
    """One figure per scan: columns = input scan, every prediction and (optionally) the reference map;
    rows = 3D view and top view of the layer up to 3 m above the terrain."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    p0 = preds[0]
    grid, ground = p0.grid, p0.ground
    sc, sl = p0.scan_coords, p0.scan_labels
    g_scan = sc[sl == G.GROUND]
    s_scan = sc[sl == G.STEM]
    gz = (np.median(ground) + 0.5) * G.DZ + G.Z_MIN
    zlim = (gz - 0.5, gz + 4.0)
    ncol = 1 + len(preds) + (reference_cells is not None)
    fig = plt.figure(figsize=(2.6 * ncol, 5.4))
    hid = np.argwhere(p0.hidden)
    hid = hid[(_height(hid, ground, grid) > 0.1) & (_height(hid, ground, grid) < 3.0)]
    hid_xy = np.unique(np.round(G.centres(hid, grid)[:, :2], 1), axis=0) if len(hid) else None

    def scan_layers(dim=1.0):
        return [(G.centres(_layer(g_scan, ground, grid), grid), C_GROUND, 0.25 * dim),
                (G.centres(_layer(s_scan, ground, grid), grid), C_STEM, 0.9 * dim)]

    cols = [('Single scan (input)', scan_layers(), f'{len(sc):,} cells')]
    for p in preds:
        g_p = _ground_on(grid, ground, p.grid)
        done = _layer(p.completed(), g_p, p.grid, 0.1, 3.0)
        cols.append((NAMES.get(p.model, p.model), scan_layers() + [(G.centres(done, p.grid), C_DONE, 0.9)],
                     f'{len(done):,} completed cells'))
    if reference_cells is not None:
        r = _layer(reference_cells, ground, grid, 0.1, 3.0)
        cols.append(('Map of all walks (reference)', scan_layers() + [(G.centres(r, grid), C_MAP, 0.5)],
                     f'{len(r):,} map cells'))
    for j, (name, layers, sub) in enumerate(cols):
        ax = fig.add_subplot(2, ncol, j + 1, projection='3d', computed_zorder=False)
        _scatter3d(ax, layers, zlim)
        ax.set_title(f'{name}\n{sub}', fontsize=8)
        ax2 = fig.add_subplot(2, ncol, ncol + j + 1)
        top = [(xyz, c, s * 1.5) for xyz, c, s in layers if c != C_GROUND]
        _top(ax2, top, hid_xy if j > 0 else None)
    handles = [Patch(color=C_GROUND, label='scan ground'), Patch(color=C_STEM, label='scan stem'),
               Patch(color=C_DONE, label='completed (hidden, 0.1-3 m above terrain)'),
               Patch(color=C_HIDDEN, label='hidden from the scan (top view)')]
    if reference_cells is not None:
        handles.append(Patch(color=C_MAP, label='map of all walks'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=7, frameon=False)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.07, wspace=0.02, hspace=0.05)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
