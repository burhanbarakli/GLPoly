"""Pattern Attention Module (PAM), makale §3.2.2, Denklem (8)–(9)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ECA


class PAM(nn.Module):
    def __init__(self, channels, n_gabor, refine=True):
        super().__init__()
        self.reducer = nn.Conv2d(n_gabor, 1, kernel_size=1)          # W_r
        self.texture = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.eca = ECA(channels) if refine else nn.Identity()    # refine=False: ablation (ECA yok)

    def forward(self, e, gabor):
        g = F.interpolate(gabor, size=e.shape[2:], mode='bilinear', align_corners=False)
        a_texture = torch.sigmoid(self.reducer(g))   # 1 × H_i × W_i
        z = self.texture(e * a_texture)
        return self.eca(z)
