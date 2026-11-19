"""Downloading the forest walks dataset (Zenodo) and the pretrained segmentation network."""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

from .segment import CKPT_URL

RECORDS = {'part1': '24378952', 'part2': '24378954'}
DOIS = {k: f'10.5281/zenodo.{v}' for k, v in RECORDS.items()}
COMPANION = 'walks_2026-08-12_companion.zip'
SEGMENTER_SHA256 = 'a9aded92b320f7d45808e5cf8d51dcff56bd5964222a6b1f064885147251a345'
CACHE = Path.home() / '.cache' / 'fosconet'
UA = {'User-Agent': 'fosconet-downloader'}


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def _hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def fetch(url: str, out: Path, size: int | None = None, checksum: str | None = None, log=print) -> Path:
    """Download url to out, resuming a partial file; checksum 'md5:...' or 'sha256:...' is verified."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and (size is None or out.stat().st_size == size):
        if checksum is None or _hash(out, checksum.split(':')[0]) == checksum.split(':')[1]:
            log(f'{out.name}: present')
            return out
    part = out.with_name(out.name + '.part')
    done = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={**UA, **({'Range': f'bytes={done}-'} if done else {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        if done and r.status != 206:
            done = 0
        total = size or (done + int(r.headers.get('Content-Length', 0)))
        with open(part, 'ab' if done else 'wb') as f:
            n, step = done, 0
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
                n += len(b)
                if total and n * 20 // total > step:
                    step = n * 20 // total
                    sys.stderr.write(f'\r{out.name}: {n / 1e9:.2f} / {total / 1e9:.2f} GB')
        sys.stderr.write('\n')
    if checksum:
        algo, ref = checksum.split(':', 1)
        got = _hash(part, algo)
        if got != ref:
            raise IOError(f'{out.name}: {algo} {got} does not match {ref}; delete {part} and retry')
    part.replace(out)
    log(f'{out.name}: downloaded')
    return out


def record_files(record_id: str) -> list[dict]:
    """Files of a published Zenodo record: key, size, checksum and download link."""
    d = _get_json(f'https://zenodo.org/api/records/{record_id}')
    files = d['files']
    if isinstance(files, dict):            # newer API layout
        files = files.get('entries', files)
        files = list(files.values()) if isinstance(files, dict) else files
    out = []
    for f in files:
        key = f.get('key') or f.get('filename')
        link = f.get('links', {}).get('self') or f.get('links', {}).get('content')
        if link and not link.endswith('/content') and '/api/records/' in link:
            link = link + '/content'
        out.append(dict(key=key, size=f.get('size') or f.get('filesize'), checksum=f.get('checksum'),
                        url=link or f'https://zenodo.org/records/{record_id}/files/{key}?download=1'))
    return out


def dataset(what: str, out: Path, log=print) -> Path:
    """Download part1, part2 or the 60 s sample (from the companion archive) into out. Returns the bag folder."""
    out = Path(out)
    if what == 'sample':
        files = {f['key']: f for f in record_files(RECORDS['part2'])}
        f = files[COMPANION]
        z = fetch(f['url'], out / COMPANION, f['size'], f['checksum'], log)
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if n.startswith('sample/') or n.startswith('map/') or n == 'README.md']
            zf.extractall(out / 'companion', names)
        bag = out / 'companion' / 'sample' / 'walks_2026-08-12_sample60s'
        log(f'sample bag: {bag}')
        return bag
    if what not in RECORDS:
        raise ValueError(f'choose sample, {", ".join(RECORDS)}')
    bag = out / f'walks_2026-08-12_{what}'
    for f in record_files(RECORDS[what]):
        if f['key'].endswith('.mcap') or f['key'] == 'metadata.yaml':
            fetch(f['url'], bag / f['key'], f['size'], f['checksum'], log)
        else:
            fetch(f['url'], out / f['key'], f['size'], f['checksum'], log)
    log(f'bag: {bag}')
    return bag


def segmenter(out: Path | None = None, log=print) -> Path:
    """The pretrained DigiForests forest_pan_seg checkpoint (Malladi et al., 2025)."""
    path = Path(out) if out else CACHE / 'forest_pan_seg_model.ckpt'
    fetch(CKPT_URL, path, None, f'sha256:{SEGMENTER_SHA256}', log)
    return path


def default_segmenter() -> Path:
    return CACHE / 'forest_pan_seg_model.ckpt'



