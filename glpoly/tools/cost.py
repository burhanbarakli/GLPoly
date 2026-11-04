"""Hesaplama maliyeti: parametre sayısı, FLOP (352×352, batch 1) ve çıkarım hızı (FPS) → revizyon/maliyet.md.

FLOP, torch.utils.flop_counter ile sayılır (çarpma-toplama = 2 FLOP); literatürde çoğunlukla verilen "GFLOPs"
aslında GMAC'tır, bu yüzden ikisi de yazılır. FPS: batch 1, eval kipi, 50 ısınma + 200 ölçüm, CUDA senkron.
FLOP'a sabit Laplace/Gabor süzgeçleri ve tüm başlıklar (5 maske + 5 sınır; testte yalnızca MP1 gerekir) dahildir;
yalnızca conv/matmul sayılır (BN, LN, aktivasyon, interpolasyon hariç, yaygın kural). Parametreye sabit süzgeçler
(buffer) dahil değildir. Ölçüm GPU boşken yapılmalı; batch 1'de süre çekirdek başlatma yüküne bağlıdır.

    python tools/cost.py
"""
import os
import sys
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.glpoly import glpoly_from_args  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'revizyon', 'maliyet.md')
CONFIGS = [
    ('GLPoly (ConvNeXt-T)', {'backbone': 'convnext_tiny', 'no_dec_norm': False}),
    ('GLPoly (Res2Net-50)', {'backbone': 'res2net50', 'no_dec_norm': False}),
    ('Baseline (ConvNeXt-T, HFAM ×, PAM ×)', {'backbone': 'convnext_tiny', 'no_dec_norm': False,
                                             'no_hfam': True, 'no_pam': True}),
    ('+PAM (ConvNeXt-T)', {'backbone': 'convnext_tiny', 'no_dec_norm': False, 'no_hfam': True}),
    ('+HFAM (ConvNeXt-T)', {'backbone': 'convnext_tiny', 'no_dec_norm': False, 'no_pam': True}),
]
SIZE = 352


def fps(model, x, dtype):
    with torch.no_grad(), torch.autocast('cuda', dtype=dtype or torch.float32, enabled=dtype is not None):
        for _ in range(50):
            model(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(200):
            model(x)
        torch.cuda.synchronize()
    return 200 / (time.perf_counter() - t)


def main():
    torch.backends.cudnn.benchmark = True
    dev = torch.device('cuda')
    rows = []
    for name, a in CONFIGS:
        m = glpoly_from_args(a, pretrained=False).eval()
        params = sum(p.numel() for p in m.parameters()) / 1e6
        x = torch.randn(1, 3, SIZE, SIZE)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            m(x)
        gflops = fc.get_total_flops() / 1e9
        m = m.to(dev)
        xg = x.to(dev)
        torch.cuda.reset_peak_memory_stats()
        f32 = fps(m, xg, None)
        mem = torch.cuda.max_memory_allocated() / 2 ** 20      # fp32 çıkarım: ağırlıklar + aktivasyonlar
        bf16 = fps(m, xg, torch.bfloat16)
        rows.append((name, params, gflops, f32, bf16, mem))
        print(f'{name}: {params:.2f}M, {gflops:.1f} GFLOP, fp32 {f32:.1f} FPS, bf16 {bf16:.1f} FPS')
        del m
        torch.cuda.empty_cache()
    lines = ['# Hesaplama maliyeti', '',
             f'`glpoly/tools/cost.py`; {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}, giriş {SIZE}×{SIZE}, batch 1.',
             'GMAC = GFLOP / 2 (literatürdeki "GFLOPs" değerleri çoğunlukla GMAC\'tır).', '',
             'Bellek: fp32 çıkarımda en yüksek ayrılan GPU belleği (ağırlıklar + aktivasyonlar + cuDNN çalışma alanı).',
             'bf16 = autocast (dönüştürülmüş model değil); batch 1\'de hızlanma beklenmeyebilir.', '',
             '| Model | Parametre (M) | GFLOP | GMAC | FPS fp32 | FPS bf16 | En yüksek bellek (MB) |', '|---|---|---|---|---|---|---|']
    for name, p, g, f32, bf, mem in rows:
        lines.append(f'| {name} | {p:.2f} | {g:.1f} | {g / 2:.1f} | {f32:.1f} | {bf:.1f} | {mem:.0f} |')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'yazıldı: {OUT}')


if __name__ == '__main__':
    main()
