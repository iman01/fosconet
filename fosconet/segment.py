"""Ground / stem labels of the scan cells from the DigiForests forest panoptic segmentation network.

FOSCo-Net was trained on scans reduced to ground and stem cells, labelled by the pretrained ``forest_pan_seg``
network of the DigiForests devkit (Malladi et al., 2025; MIT licence, model files in ``third_party/digiforests``).
The network runs on windows of 9 consecutive scans in the map frame with 0.1 m voxels, and the class
probabilities of every voxel are summed over the windows. The labels of a scan come from the windows that start
within the past 5 s (the window holding the scan included), i.e. what a robot has while it drives.

Requires MinkowskiEngine and the pretrained checkpoint (``fosconet download segmenter``).
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

CKPT_URL = 'http://ipb.uni-bonn.de/html/deeplearningmodels/malladi2025icra/forest_pan_seg_model.ckpt'
VDS = 0.1             # voxel size of the segmenter [m]
STRIDE = 16           # deepest tensor stride of MinkUNet14: window coordinates are aligned to it
REFLECTIVITY_GAIN = 1.6   # Ouster calibrated reflectivity -> the intensity scale of the DigiForests training data
WINDOW = 9            # scans per segmentation window
PAST_S = 5.0          # s of past windows voting for a scan
RATE_HZ = 20.0
_OFF = 1 << 20


def _pack(k: torch.Tensor) -> torch.Tensor:
    return ((k[:, 0] + _OFF) << 42) | ((k[:, 1] + _OFF) << 21) | (k[:, 2] + _OFF)


def _unpack(key: torch.Tensor) -> torch.Tensor:
    m = (1 << 21) - 1
    return torch.stack([((key >> 42) & m) - _OFF, ((key >> 21) & m) - _OFF, (key & m) - _OFF], 1)


def voxel_keys(xyz: np.ndarray, device) -> torch.Tensor:
    return _pack(torch.floor(torch.as_tensor(xyz, dtype=torch.float64, device=device) / VDS).long())


class Segmenter:
    """Semantic head of the pretrained forest_pan_seg network (classes ignore, ground, shrub, stem, canopy)."""

    def __init__(self, checkpoint, device: str | None = None):
        try:
            import MinkowskiEngine as ME  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError('labelling the scans needs MinkowskiEngine, see docs/INSTALL.md') from e
        from .third_party.digiforests.minkunet import CustomMinkUNet14
        from .third_party.digiforests.mlp import MLP
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f'{path} not found: run "fosconet download segmenter" or get it from {CKPT_URL}')
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        ck = torch.load(path, map_location='cpu', weights_only=True)
        hp = ck['hyper_parameters']
        if abs(float(hp['vds']) - VDS) > 1e-9:
            raise ValueError(f'unexpected segmenter voxel size {hp["vds"]}')
        self.unet = CustomMinkUNet14(in_channels=hp['in_channels'], out_channels=hp['embedding_size'],
                                     D=hp['coord_dimension'])
        self.head = MLP(hp['embedding_size'], hp['num_classes'], batch_norm=True, num_layers=2)
        sd = ck['state_dict']
        self.unet.load_state_dict({k[10:]: v for k, v in sd.items() if k.startswith('mink_unet.')})
        self.head.load_state_dict({k[18:]: v for k, v in sd.items() if k.startswith('segmentation_head.')})
        self.unet.to(self.device).eval()
        self.head.to(self.device).eval()
        self.num_classes = int(hp['num_classes'])

    @torch.no_grad()
    def predict(self, xyz: np.ndarray, intensity: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Points of one window (map frame) -> sorted unique voxel keys (M,) and class probabilities (M, C)."""
        import MinkowskiEngine as ME
        key = voxel_keys(xyz, self.device)
        uk, inv = torch.unique(key, return_inverse=True)
        cnt = torch.bincount(inv, minlength=len(uk)).float()
        f = torch.zeros(len(uk), device=self.device).index_add_(
            0, inv, torch.as_tensor(intensity, dtype=torch.float32, device=self.device)) / cnt
        k3 = _unpack(uk)
        base = (k3.median(0).values // STRIDE) * STRIDE
        c = (k3 - base).int()
        bcol = torch.zeros((len(c), 1), dtype=torch.int32, device=self.device)
        out = self.unet(ME.SparseTensor(f.view(-1, 1), torch.cat([bcol, c], 1)))
        return uk, torch.softmax(self.head(out.F), 1).half()


class PastVote:
    """Window predictions over a stream of scans and the past-only vote for any scan of the stream.

    add(index, xyz_map, intensity) for every scan in order; labels(index, xyz_map) once the window holding the
    scan is complete (``ready(index)``).
    """

    def __init__(self, segmenter: Segmenter, window: int = WINDOW, past_s: float = PAST_S, rate_hz: float = RATE_HZ):
        self.seg, self.win = segmenter, window
        self.past = int(round(past_s * rate_hz))
        self.frames: OrderedDict[int, tuple] = OrderedDict()
        self.windows: OrderedDict[int, tuple] = OrderedDict()
        self.last = -1

    def add(self, index: int, xyz_map: np.ndarray, intensity: np.ndarray):
        self.frames[index] = (xyz_map.astype(np.float32), intensity.astype(np.float32))
        self.last = index
        if (index + 1) % self.win == 0:
            self._run(index + 1 - self.win)
        oldest = index - self.past - 2 * self.win
        for i in [i for i in self.frames if i < oldest]:
            del self.frames[i]
        for s in [s for s in self.windows if s < oldest]:
            del self.windows[s]

    def finish(self):
        """Run the last, incomplete window at the end of the stream."""
        s = (self.last // self.win) * self.win
        if s not in self.windows and any(i >= s for i in self.frames):
            self._run(s)

    def _run(self, s: int):
        fr = [self.frames[i] for i in range(s, s + self.win) if i in self.frames]
        if fr:
            self.windows[s] = self.seg.predict(np.concatenate([f[0] for f in fr]), np.concatenate([f[1] for f in fr]))

    def ready(self, index: int) -> bool:
        return (index // self.win) * self.win in self.windows

    def labels(self, index: int, xyz_map: np.ndarray) -> np.ndarray:
        """Class of every point of scan ``index`` from the windows starting within the past ``past_s``."""
        s_own = (index // self.win) * self.win
        use = [s for s in self.windows if index - self.past <= s <= s_own]
        K = torch.cat([self.windows[s][0] for s in use])
        P = torch.cat([self.windows[s][1] for s in use]).float()
        uk, inv = torch.unique(K, return_inverse=True)
        acc = torch.zeros((len(uk), P.shape[1]), device=P.device).index_add_(0, inv, P)
        pk = voxel_keys(xyz_map, P.device)
        idx = torch.searchsorted(uk, pk).clamp(max=len(uk) - 1)
        lab = acc[idx].argmax(1)
        lab[uk[idx] != pk] = 0
        return lab.byte().cpu().numpy()
