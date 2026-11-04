"""ECA ve Spatial Attention (makale Şekil: ECA → SA sıralı yapı)."""
import math

import torch
import torch.nn as nn


class ECA(nn.Module):
    """Efficient Channel Attention: GAP → Conv1d(k) → sigmoid, boyut indirgemesiz.

    k, ECA-Net'teki gibi kanal sayısından uyarlamalı seçilir (gamma=2, b=1).
    """

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.k = k
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        s = self.pool(x)                                  # B,C,1,1   (s_c)
        w = self.conv(s.squeeze(-1).transpose(-1, -2))    # B,1,C
        w = torch.sigmoid(w.transpose(-1, -2).unsqueeze(-1))
        return x * w


class SpatialAttention(nn.Module):
    """A = [Avg_c(I'); Max_c(I')] → Conv7x7 → sigmoid."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        a = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(a))


class ECASpatial(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.eca = ECA(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.eca(x))
