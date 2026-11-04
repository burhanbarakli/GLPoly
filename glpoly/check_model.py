"""Hızlı doğrulama: şekiller, ablation varyantları, kayıp/geri yayılım, parametre grupları, MPS."""
import sys
import time

import torch

from lib.glpoly import GLPoly, param_groups
from utils.losses import GLPolyLoss

device = torch.device('cuda' if torch.cuda.is_available() else
                      ('mps' if torch.backends.mps.is_available() else 'cpu'))


def sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()



speed = '--speed' in sys.argv
backbones = [a for a in sys.argv[1:] if not a.startswith('--')] or ['res2net50', 'convnext_tiny']

for bb in backbones:
    for hfam, pam in [(True, True), (False, False), (False, True), (True, False)]:
        m = GLPoly(bb, pretrained=(hfam and pam), use_hfam=hfam, use_pam=pam).to(device)
        x = torch.randn(2, 3, 352, 352, device=device)
        masks, bounds = m(x)
        shapes = [tuple(t.shape[-2:]) for t in masks]
        assert shapes == [(352, 352), (176, 176), (88, 88), (44, 44), (22, 22)], shapes
        assert [t.shape for t in bounds] == [t.shape for t in masks]
        y = (torch.rand(2, 1, 352, 352, device=device) > 0.7).float()
        loss, lm, lb = GLPolyLoss()(masks, bounds, y, y)
        loss.backward()
        no_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        assert not no_grad, no_grad[:5]
        n = sum(p.numel() for p in m.parameters()) / 1e6
        print(f'{bb:14s} hfam={hfam!s:5} pam={pam!s:5} params={n:6.2f}M loss={loss.item():.3f} OK', flush=True)
        del m, masks, bounds, loss
        if device.type == 'mps':
            torch.mps.empty_cache()

    m = GLPoly(bb, pretrained=False).to(device)
    groups = param_groups(m, 0.33, 1e-4)
    total = sum(p.numel() for g in groups for p in g['params'])
    assert total == sum(p.numel() for p in m.parameters())
    print('  param groups:', [(g['lr_mult'], g['weight_decay'], sum(p.numel() for p in g['params']))
                              for g in groups])

    del m, groups
    if not speed:
        continue
    # hız: batch 4 (16 GB'ta batch 8 × 448 px belleğe sığmıyor), 3 ölçek, ileri+geri
    m = GLPoly(bb, pretrained=False).to(device).train()
    opt = torch.optim.AdamW(m.parameters(), 1e-4)
    for size in (256, 352, 448):
        x = torch.randn(4, 3, size, size, device=device)
        y = (torch.rand(4, 1, size, size, device=device) > 0.7).float()
        for k in range(3):
            if k == 1:
                sync()
                t0 = time.time()
            masks, bounds = m(x)
            loss = GLPolyLoss()(masks, bounds, y, y)[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
        sync()
        print(f'  {size}px batch 4: {(time.time() - t0) / 2:.2f} s/iter', flush=True)
    if device.type == 'mps':
        print(f'  MPS bellek: {torch.mps.driver_allocated_memory() / 2**30:.1f} GB')
