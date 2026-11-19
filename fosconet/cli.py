"""Command line interface: ``fosconet <command> --help``."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _models(s: str) -> list[str]:
    return [m.strip() for m in s.split(',') if m.strip()]


def cmd_download(a):
    from . import download as D
    if a.what == 'segmenter':
        D.segmenter(a.out)
    else:
        D.dataset(a.what, Path(a.out or 'data'))


def _segmenter(path):
    from . import download as D
    from .segment import Segmenter
    p = Path(path) if path else D.default_segmenter()
    if not p.exists():
        print(f'segmentation checkpoint not found at {p}: downloading it', file=sys.stderr)
        p = D.segmenter(p)
    return Segmenter(p)


def cmd_prepare(a):
    from .bag import Recording
    from .pipeline import prepare
    rec = Recording(a.bag, point_stride=a.point_stride)
    out = Path(a.out)
    t0, n = time.time(), 0
    for s in prepare(rec, _segmenter(a.segmenter), tuple(a.grid), a.every, a.start, a.end, not a.no_deskew,
                     log=lambda *x: None):
        s.save(out / f'{s.name}.npz', keep_points=a.keep_points)
        n += 1
        c, lb = s.cells[a.grid[0]]
        print(f'{s.name}: {len(c):,} cells ({int((lb == 3).sum()):,} stem)  [{time.time() - t0:.0f} s]', flush=True)
    print(f'{n} samples in {out}')


def _sample_files(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob('*.npz')) if p.is_dir() else [p]
    return files


def cmd_predict(a):
    from .io import save_prediction
    from .pipeline import Sample
    from .predict import Predictor
    files = _sample_files(a.samples)
    out = Path(a.out)
    for m in _models(a.model):
        pr = Predictor(m, device=a.device)
        times = []
        for f in files:
            s = Sample.load(f)
            if pr.grid not in s.cells:
                raise SystemExit(f'{f} has no {pr.grid} grid: run prepare with --grid {pr.grid}')
            c, lb = s.cells[pr.grid]
            res = pr(c, lb, s.vis[pr.grid])
            times.append(res.seconds)
            save_prediction(out / m / f.name, s, pr.grid, res, m)
        print(f'{m}: {len(files)} scans -> {out / m} (median {1e3 * np.median(times):.0f} ms per scan, '
              f'threshold {pr.threshold})')
        if a.ply:
            from .io import Prediction, export_ply
            for f in files:
                export_ply(Prediction(out / m / f.name), (out / m / f.name).with_suffix('.ply'), a.frame)


def cmd_render(a):
    from .io import Prediction
    from .render import figure
    preds = [Prediction(p) for p in a.predictions]
    ref = None
    if a.map:
        from .reference import MapReference
        ref = MapReference(a.map, preds[0].T_map_ego[:3, 3][None]).cells(preds[0].T_map_ego, preds[0].grid)
    out = a.out or Path(a.predictions[0]).with_suffix('.png')
    figure(preds, out, ref, a.title)
    print(out)


def cmd_run(a):
    """prepare + predict + render for a recording."""
    out = Path(a.out)
    grids = sorted({'polar' if m.endswith('polar') else 'cart' for m in _models(a.model)})
    cmd_prepare(argparse.Namespace(bag=a.bag, out=out / 'samples', grid=grids, every=a.every, start=a.start,
                                   end=a.end, no_deskew=False, keep_points=False, segmenter=a.segmenter,
                                   point_stride=a.point_stride))
    cmd_predict(argparse.Namespace(samples=[out / 'samples'], model=a.model, out=out / 'predictions', device=None,
                                   ply=a.ply, frame='map'))
    if a.figures:
        files = _sample_files([out / 'samples'])[:a.figures]
        for f in files:
            preds = [str(out / 'predictions' / m / f.name) for m in _models(a.model)]
            cmd_render(argparse.Namespace(predictions=preds, out=out / 'figures' / f.with_suffix('.png').name,
                                          map=a.map, title=None))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='fosconet', description='FOSCo-Net: completion of trunk-occluded geometry in '
                                 'single forest LiDAR scans.')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('download', help='download the dataset (sample, part1, part2) or the segmentation network')
    p.add_argument('what', choices=['sample', 'part1', 'part2', 'segmenter'])
    p.add_argument('--out', default=None, help='folder (dataset, default ./data) or file (segmenter)')
    p.set_defaults(func=cmd_download)

    def bag_args(p):
        p.add_argument('--every', type=int, default=20, help='use every n-th scan (default 20: one per second)')
        p.add_argument('--start', type=float, default=None, help='seconds from the first scan')
        p.add_argument('--end', type=float, default=None, help='seconds from the first scan')
        p.add_argument('--segmenter', default=None, help='forest_pan_seg checkpoint (default: downloaded once)')
        p.add_argument('--point-stride', type=int, default=3,
                       help='keep every n-th raw point (3 reproduces the density of the training scans)')

    p = sub.add_parser('prepare', help='scans of a recording -> labelled grids with visibility (one .npz per scan)')
    p.add_argument('bag', help='bag folder (the .mcap file and metadata.yaml)')
    p.add_argument('--out', required=True)
    p.add_argument('--grid', nargs='+', default=['polar'], choices=['polar', 'cart'])
    p.add_argument('--no-deskew', action='store_true')
    p.add_argument('--keep-points', action='store_true', help='also store the labelled points of every scan')
    bag_args(p)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser('predict', help='complete prepared scans')
    p.add_argument('samples', nargs='+', help='sample files or folders from prepare')
    p.add_argument('--model', default='fosconet2-polar',
                   help='comma list of fosconet2-polar, fosconet2-cart, fosconet1-polar, fosconet1-cart or .pt files')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default=None)
    p.add_argument('--ply', action='store_true', help='also write a coloured point cloud per scan')
    p.add_argument('--frame', default='ego', choices=['ego', 'map'], help='frame of the .ply files')
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser('render', help='figure of one scan: input, predictions, optional map reference')
    p.add_argument('predictions', nargs='+', help='prediction files of the same scan (one per model)')
    p.add_argument('--out', default=None)
    p.add_argument('--map', default=None, help='map/walks_2026-08-12_map_local_enu.pcd of the companion archive')
    p.add_argument('--title', default=None)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser('run', help='prepare, predict and render a recording')
    p.add_argument('bag')
    p.add_argument('--out', required=True)
    p.add_argument('--model', default='fosconet2-polar,fosconet1-polar')
    p.add_argument('--figures', type=int, default=3, help='render the first n scans')
    p.add_argument('--map', default=None, help='map of all walks, as a reference in the figures')
    p.add_argument('--ply', action='store_true')
    bag_args(p)
    p.set_defaults(func=cmd_run)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == '__main__':
    main()
