"""High-Frequency Attention Module (HFAM), makale §3.2.1.

BB-Res2Net_orj'daki EGA modülünden türetilmiştir; farklar:
  * kenar girdisi yalnızca Laplace haritası L_i (Gabor ile birleştirilmez),
  * CBAM yerine ECA → Spatial Attention,
  * EGA'daki `+ E_i` artık bağlantısı makalede (F_c denklemi ve şekil) yok; varsayılan kapalı,
    yalnızca ek deney için `--hfam_residual` ile açılır.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ECASpatial
from .cues import make_laplace


PARTS = ('bg', 'bdy', 'edge')


def corrupt_mask(m, p):
    """Yalnızca eğitimde: her örnekte p olasılıkla kaba maskeyi rastgele aşındırır ya da genişletir
    (revizyon/darbogaz_analizi.md, D3: başarısız görüntülerde kaba maske polibin ancak %31–41'ini kapsıyor)."""
    out = []
    for mi in m.split(1):
        if float(torch.rand(())) < p:
            k = 2 * int(torch.randint(1, max(1, mi.shape[-1] // 16) + 1, ())) + 1   # yarıçap ≤ kenarın ~1/16'sı
            mi = F.max_pool2d(mi, k, 1, k // 2) if float(torch.rand(())) < 0.5 else -F.max_pool2d(-mi, k, 1, k // 2)
        out.append(mi)
    return torch.cat(out)


class HFAM(nn.Module):
    """parts: kullanılan dallar (ablation için bir dal çıkarılabilir); refine=False: ECA+SA yok."""

    def __init__(self, channels, residual=False, parts=PARTS, refine=True):
        super().__init__()
        assert parts and set(parts) <= set(PARTS), parts
        self.residual, self.parts = residual, tuple(p for p in PARTS if p in parts)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(self.parts), channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.refine = ECASpatial(channels) if refine else nn.Identity()
        self.corrupt = 0.0                                   # eğitimde maske bozma olasılığı (train.py --mask_corrupt)

    def forward(self, e, prev_mask, edge):
        size = e.shape[2:]
        with torch.autocast(e.device.type, enabled=False):   # Δ ve dikkat haritaları fp32'de
            m = torch.sigmoid(F.interpolate(prev_mask.float(), size=size, mode='bilinear', align_corners=False))
            if self.training and self.corrupt > 0:
                m = corrupt_mask(m, self.corrupt)
            l = F.interpolate(edge.float(), size=size, mode='bilinear', align_corners=False) \
                if 'edge' in self.parts else None
            bdy = make_laplace(m, 1)

        branch = {'bg': lambda: e * (1 - m).to(e.dtype),     # ters dikkat (arka plan)
                  'bdy': lambda: e * bdy.to(e.dtype),        # Δ(σ(M)) – sınır dikkati
                  'edge': lambda: e * l.to(e.dtype)}         # görüntüden gelen yüksek frekans

        f_c = self.fuse(torch.cat([branch[p]() for p in self.parts], dim=1))
        if self.residual:
            f_c = f_c + e
        return self.refine(f_c)
