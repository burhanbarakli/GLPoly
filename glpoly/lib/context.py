"""/32 seviyesinde genel bağlam bloğu (revizyon/darbogaz_analizi.md, D1/D5).

İnce ayarlı ConvNeXt'in /32 etkin alıcı alanı büyük poliplerden küçük kalıyor ve kaba maske (d5) bu seviyeden
üretiliyor. Bu blok /32 özelliklerinin tamamı üzerinde tek bir transformer katmanı çalıştırır: koşullu konum
kodlaması (3×3 derinlemesine conv, PVTv2/CPVT gibi; çok ölçekli eğitimde boyuttan bağımsız) + MHSA + MLP.
Artık dallar küçük bir LayerScale ile başlar; blok başlangıçta neredeyse birim dönüşümdür.
"""
import torch
import torch.nn as nn


class GlobalContext(nn.Module):
    def __init__(self, channels, heads=8, mlp_ratio=4, init_scale=0.1):
        super().__init__()
        self.pos = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * mlp_ratio), nn.GELU(),
                                 nn.Linear(channels * mlp_ratio, channels))
        self.gamma1 = nn.Parameter(torch.full((channels,), init_scale))
        self.gamma2 = nn.Parameter(torch.full((channels,), init_scale))

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.pos(x)
        t = x.flatten(2).transpose(1, 2)                      # B × HW × C
        y = self.norm1(t)
        t = t + self.gamma1 * self.attn(y, y, y, need_weights=False)[0]
        t = t + self.gamma2 * self.mlp(self.norm2(t))
        return t.transpose(1, 2).reshape(b, c, h, w)
