"""The FOSCo-Net network.

A sparse 3D encoder (submanifold sparse convolutions, three stride-2 stages) processes the occupied cells of the
single scan. A dense decoder restores the full grid by trilinear upsampling and convolutions, receiving the encoder
features through sigmoid-gated skips

    d_i <- d_i + sigmoid(W_i d_i) * e_i ,

which vanish where the scan has no cell. FOSCo-Net 2.0 adds the scan's visibility (seen, hidden) and the height
above the terrain as dense channels at every decoder stage, the height as a fourth input feature, and pads the
azimuth axis of the polar grid with 64 copied cells before the encoder (exact equivariance to rotations by
multiples of 8 deg). FOSCo-Net 1.0 is the same network without these additions.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import spconv.pytorch as spconv
except ImportError as e:  # pragma: no cover
    raise ImportError('FOSCo-Net needs spconv 2.x: pip install spconv-cu124 (pick the build of your CUDA version, '
                      'see https://github.com/traveller59/spconv)') from e

from . import grid as G

SEAM = 64          # azimuth cells copied on each side of the polar grid before the encoder


@dataclass
class Config:
    """Inputs and options of a FOSCo-Net model (stored with its weights)."""
    grid: str = 'polar'
    visibility: bool = True     # seen / hidden dense channels
    height: bool = True         # height above the terrain: input feature and dense channel
    seam: bool = True           # azimuth padding (polar grid only)
    labels: bool = True         # ground / stem indicators as input features
    base_channels: int = 16

    @property
    def dense_channels(self) -> int:
        return 2 * self.visibility + self.height

    @property
    def in_channels(self) -> int:
        return 3 + self.height

    @classmethod
    def from_dict(cls, d: dict) -> 'Config':
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


class SparseResBlock(spconv.SparseModule):
    """Two submanifold convolutions with batch norm, ReLU and a residual connection."""

    def __init__(self, in_ch: int, out_ch: int, indice_key: str):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(out_ch, out_ch, 3, padding=1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = (spconv.SubMConv3d(in_ch, out_ch, 1, bias=False, indice_key=f'{indice_key}_skip')
                     if in_ch != out_ch else None)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.net(x)
        skip = self.skip(x).features if self.skip is not None else x.features
        return res.replace_feature(self.relu(res.features + skip))


class Up(nn.Module):
    """Trilinear upsampling to a given size, then two 3 x 3 x 3 convolutions with batch norm and ReLU. The azimuth
    axis is padded circularly on the polar grid without seam padding, otherwise with zeros."""

    def __init__(self, cin: int, cout: int, circular: bool):
        super().__init__()
        self.circular = circular
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(cout)

    def _pad(self, x):
        if self.circular:
            x = F.pad(x, [1, 1, 0, 0, 1, 1])
            return torch.cat([x[:, :, :, -1:, :], x, x[:, :, :, :1, :]], dim=3)
        return F.pad(x, [1, 1, 1, 1, 1, 1])

    def forward(self, x, size=None):
        if size is not None:
            x = F.interpolate(x, size=tuple(size), mode='trilinear', align_corners=False)
        x = F.relu(self.bn1(self.conv1(self._pad(x))), inplace=True)
        return F.relu(self.bn2(self.conv2(self._pad(x))), inplace=True)


class FOSCoNet(nn.Module):
    """forward(coords (N, 4) int32 [batch, a, b, k], feats (N, C), batch_size, dense (B, D, *grid)) -> logits
    (B, 1, *grid)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        c = cfg.base_channels
        self.seam = SEAM if (cfg.seam and cfg.grid == 'polar') else 0
        circ = cfg.grid == 'polar' and not self.seam
        self.enc0 = SparseResBlock(cfg.in_channels, c, 'enc0')
        self.down0 = spconv.SparseConv3d(c, c * 2, 3, stride=2, padding=1, bias=False, indice_key='down0')
        self.enc1 = SparseResBlock(c * 2, c * 2, 'enc1')
        self.down1 = spconv.SparseConv3d(c * 2, c * 4, 3, stride=2, padding=1, bias=False, indice_key='down1')
        self.enc2 = SparseResBlock(c * 4, c * 4, 'enc2')
        self.down2 = spconv.SparseConv3d(c * 4, c * 8, 3, stride=2, padding=1, bias=False, indice_key='down2')
        self.bottleneck = SparseResBlock(c * 8, c * 8, 'btn')
        self.dec2, self.dec1, self.dec0 = Up(c * 8, c * 4, circ), Up(c * 4, c * 2, circ), Up(c * 2, c, circ)
        self.gate2, self.gate1, self.gate0 = nn.Conv3d(c * 4, c * 4, 1), nn.Conv3d(c * 2, c * 2, 1), nn.Conv3d(c, c, 1)
        dc = cfg.dense_channels
        if dc:
            self.inj2, self.inj1, self.inj0 = nn.Conv3d(dc, c * 4, 1), nn.Conv3d(dc, c * 2, 1), nn.Conv3d(dc, c, 1)
        self.head = nn.Conv3d(c, 1, 1)

    @staticmethod
    def _skip(x, e, gate):
        return x + torch.sigmoid(gate(x)) * e

    def _wrap(self, t, dim):
        S = self.seam
        return torch.cat([t.narrow(dim, t.shape[dim] - S, S), t, t.narrow(dim, 0, S)], dim=dim)

    def forward(self, coords, feats, batch_size: int, dense=None):
        S, shape = self.seam, list(G.SHAPE[self.cfg.grid])
        if S:
            nt = shape[1]
            th = coords[:, 2]
            lo, hi = th < S, th >= nt - S
            c0 = coords.clone()
            c0[:, 2] += S
            cl = coords[lo].clone()
            cl[:, 2] += nt + S
            ch = coords[hi].clone()
            ch[:, 2] -= nt - S
            coords = torch.cat([c0, cl, ch])
            feats = torch.cat([feats, feats[lo], feats[hi]])
            shape[1] = nt + 2 * S
            if dense is not None:
                dense = self._wrap(dense, 3)
        x = spconv.SparseConvTensor(features=feats, indices=coords, spatial_shape=shape, batch_size=batch_size)
        e0 = self.enc0(x)
        e1 = self.enc1(self.down0(e0))
        e2 = self.enc2(self.down1(e1))
        x = self.bottleneck(self.down2(e2)).dense()
        x = self.dec2(x, e2.spatial_shape)
        if dense is not None:
            x = x + self.inj2(F.adaptive_avg_pool3d(dense, tuple(e2.spatial_shape)))
        x = self._skip(x, e2.dense(), self.gate2)
        x = self.dec1(x, e1.spatial_shape)
        if dense is not None:
            x = x + self.inj1(F.adaptive_avg_pool3d(dense, tuple(e1.spatial_shape)))
        x = self._skip(x, e1.dense(), self.gate1)
        x = self.dec0(x, shape)
        if dense is not None:
            x = x + self.inj0(dense)
        x = self._skip(x, e0.dense(), self.gate0)
        out = self.head(x)
        return out[:, :, :, S:S + G.NT] if S else out
