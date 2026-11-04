"""Bir koşunun özet görselleri: eğitim/doğrulama eğrileri ve örnek tahminler (MP1..MP5, BP1, ikili maske).

    python tools/plot_run.py --name smoke [--ckpt best.pth]
"""
import argparse
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.glpoly import glpoly_from_args  # noqa: E402
from utils.data import PolypEval, list_pairs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EPOCH_RE = re.compile(r'EPOCH (\d+) loss ([\d.]+) val_dice ([\d.]+)@([\d.]+) \(0\.5: ([\d.]+)\)')
STEP_RE = re.compile(r'epoch (\d+)/\d+ step (\d+)/(\d+) loss ([\d.]+) \(mask ([\d.]+), bdy ([\d.]+)\) lr ([\d.e+-]+)')


def curves(run_dir, plt):
    ep, st = [], []
    with open(os.path.join(run_dir, 'log.txt'), encoding='utf-8', errors='replace') as f:
        for line in f:
            # yeniden başlatma/devam günlüğe üst üste yazar: geri giden epoch'ta öncekinin o noktadan sonrasını at
            if (m := EPOCH_RE.search(line)):
                row = [float(v) for v in m.groups()]
                ep = [r for r in ep if r[0] < row[0]] + [row]
            elif (m := STEP_RE.search(line)):
                e, s, n, *rest = m.groups()
                x = int(e) - 1 + int(s) / int(n)
                st = [r for r in st if r[0] < x] + [[x, *map(float, rest)]]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    if st:
        st = np.array(st)
        ax[0].plot(st[:, 0], st[:, 1], label='toplam')
        ax[0].plot(st[:, 0], st[:, 2], label='maske')
        ax[0].plot(st[:, 0], st[:, 3], label='sınır')
        ax[0].set_xlabel('epoch'), ax[0].set_title('eğitim kaybı (kayan ortalama)'), ax[0].legend(fontsize=8)
        ax[2].plot(st[:, 0], st[:, 4]), ax[2].set_xlabel('epoch'), ax[2].set_title('head LR')
    if ep:
        ep = np.array(ep)
        ax[1].plot(ep[:, 0], ep[:, 2], 'o-', label='en iyi eşikte')
        ax[1].plot(ep[:, 0], ep[:, 4], 's--', label='eşik 0.5')
        for x, y, t in zip(ep[:, 0], ep[:, 2], ep[:, 3]):
            ax[1].annotate(f'{t:.2f}', (x, y), fontsize=7, xytext=(2, 4), textcoords='offset points')
        ax[1].set_xlabel('epoch'), ax[1].set_title('doğrulama Dice (etiket: seçilen eşik)'), ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(run_dir, 'egriler.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def samples(run_dir, ckpt, plt, device):
    ck = torch.load(os.path.join(run_dir, ckpt), map_location='cpu', weights_only=False)
    a = ck['args']
    model = glpoly_from_args(a)
    model.load_state_dict(ck['model'])
    model.to(device).eval()
    th = ck['threshold']
    picks = [('CVC-ClinicDB', 0), ('Kvasir', 0), ('CVC-ColonDB', 52), ('ETIS-LaribPolypDB', 3)]
    cols = ['görüntü', 'GT', 'MP1', 'MP2', 'MP3', 'MP4', 'MP5', 'BP1', f'ikili (eşik {th:.2f})']
    fig, ax = plt.subplots(len(picks), len(cols), figsize=(1.9 * len(cols), 2.0 * len(picks)))
    for r, (ds, k) in enumerate(picks):
        pairs = list_pairs(os.path.join(ROOT, 'dataset', 'TestDataset', ds))
        img, gt, name = PolypEval([pairs[k]], a['trainsize'])[0]
        with torch.no_grad():
            masks, bounds = model(img[None].to(device))
        rgb = np.asarray(Image.open(pairs[k][0]).convert('RGB'))
        mp = [torch.sigmoid(m)[0, 0].cpu().numpy() for m in masks]
        bp1 = torch.sigmoid(bounds[0])[0, 0].cpu().numpy()
        prob = torch.sigmoid(F.interpolate(masks[0], size=gt.shape, mode='bilinear', align_corners=False))[0, 0]
        binary = (prob.cpu().numpy() >= th)
        g = gt.numpy()
        dice = 2 * (binary * g).sum() / (binary.sum() + g.sum() + 1e-8)
        panels = [rgb, g, *mp, bp1, binary]
        for c, im in enumerate(panels):
            ax[r, c].imshow(im, cmap=None if c == 0 else 'gray', vmin=0, vmax=1)
            ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(cols[c] + (f'\n{im.shape[0]}px' if 2 <= c <= 7 else ''), fontsize=8)
        ax[r, 0].set_ylabel(f'{ds}\n{name}', fontsize=7)
        ax[r, -1].set_xlabel(f'Dice {dice:.3f}', fontsize=8)
    fig.tight_layout()
    path = os.path.join(run_dir, 'ornek_tahminler.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--ckpt', default='best.pth')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    opt = p.parse_args()
    run_dir = os.path.join(ROOT, 'runs', opt.name)
    print(curves(run_dir, plt))
    print(samples(run_dir, opt.ckpt, plt, opt.device))


if __name__ == '__main__':
    main()
