"""İstatistikler: görüntü başına Dice, bootstrap güven aralıkları, eşli testler, polip boyutu katmanları.

Görüntü başına Dice/IoU, PraNet protokolüyle (preds_pranet/, 256 eşik ortalaması) evaluate.py ile aynı kuralla
hesaplanır (Fmeasure_calu: pred >= eşik; TP = 0 ise Dice = 0); sonuçlar runs/<ad>/perimage_pranet.csv'ye önbelleklenir.
Doğrulama: her koşu için set ortalaması results_pranet.md'deki meanDic ile karşılaştırılır.

Ablasyon koşuları revize tariftendir (D12, seed 42). Önceki tarifin sonuçları: revizyon/istatistik.md (git geçmişi).

    python tools/stats.py            # → revizyon/istatistik_d12.md
"""
import csv
import os
import sys

import numpy as np
from PIL import Image
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import IMG_EXT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS, TEST = os.path.join(ROOT, 'runs'), os.path.join(ROOT, 'dataset', 'TestDataset')
OUT = os.path.join(ROOT, 'revizyon', 'istatistik_d12.md')
SETS = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300']
UNSEEN = ['CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300']
SHORT = {'Kvasir': 'Kvasir', 'CVC-ClinicDB': 'ClinicDB', 'CVC-ColonDB': 'ColonDB', 'ETIS-LaribPolypDB': 'ETIS',
         'CVC-300': 'CVC-300'}
ABL = [('Baseline', 'abl_d12_baseline'), ('+PAM', 'abl_d12_pam'), ('+HFAM', 'abl_d12_hfam'), ('Proposed', 'ctl_d12_jit_clahe_lr01')]
THR = np.linspace(1, 0, 256)
GE = (np.arange(256)[:, None].astype(np.float64) / 255) >= THR[None, :]   # [değer k, eşik j]: k/255 >= eşik_j
N_BOOT, RNG = 10000, np.random.default_rng(0)


def image_scores(pred_u8, gt):
    """Bir görüntünün 256 eşik üzerinden ortalama Dice ve IoU'su (evaluate.py ile aynı)."""
    hist_all = np.bincount(pred_u8.ravel(), minlength=256)
    hist_pos = np.bincount(pred_u8[gt].ravel(), minlength=256)
    p = hist_all @ GE          # eşik başına tahmin pozitif sayısı
    tp = hist_pos @ GE
    g = gt.sum()
    dice = np.where(tp > 0, 2 * tp / np.maximum(g + p, 1), 0.0)
    iou = np.where(tp > 0, tp / np.maximum(g - tp + p, 1), 0.0)
    return dice.mean(), iou.mean(), g / gt.size


def per_image(run, ds):
    """{görüntü: (dice, iou, gt_oranı)}; runs/<ad>/perimage_pranet.csv önbelleği."""
    cache = os.path.join(RUNS, run, 'perimage_pranet.csv')
    rows = {}
    if os.path.exists(cache):
        with open(cache, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rows.setdefault(r['dataset'], {})[r['image']] = (float(r['dice']), float(r['iou']), float(r['area']))
    if ds not in rows:
        pdir, gdir = os.path.join(RUNS, run, 'preds_pranet', ds), os.path.join(TEST, ds, 'masks')
        out = {}
        for f in sorted(x for x in os.listdir(gdir) if x.lower().endswith(IMG_EXT)):
            stem = os.path.splitext(f)[0]
            pf = next(x for x in os.listdir(pdir) if os.path.splitext(x)[0] == stem)
            pred = np.asarray(Image.open(os.path.join(pdir, pf)).convert('L'))
            gt = np.asarray(Image.open(os.path.join(gdir, f)).convert('L')) > 127
            out[stem] = image_scores(pred, gt)
        rows[ds] = out
        with open(cache, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['dataset', 'image', 'dice', 'iou', 'area'])
            for d, imgs in rows.items():
                for k, (a, b, c) in imgs.items():
                    w.writerow([d, k, f'{a:.6f}', f'{b:.6f}', f'{c:.6f}'])
    return rows[ds]


def table_mean(run, ds):
    with open(os.path.join(RUNS, run, 'results_pranet.md'), encoding='utf-8') as f:
        for ln in f:
            if ln.startswith(ds + ' '):
                return float(ln.split()[1])
    return None


def boot_ci(x):
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def vec(run, sets):
    """Setler boyunca aynı sırada görüntü başına Dice ve alan."""
    d, a = [], []
    for ds in sets:
        pi = per_image(run, ds)
        for k in sorted(pi):
            d.append(pi[k][0])
            a.append(pi[k][2])
    return np.array(d), np.array(a)


def fmt_ci(m, lo, hi):
    return f'{m:.3f} [{lo:.3f}, {hi:.3f}]'


def main():
    runs = [r for _, r in ABL] + ['glpoly_res2net_dn']
    lines = ['# İstatistik (PraNet protokolü, görüntü başına)', '',
             '`glpoly/tools/stats.py` ile üretilir. Güven aralıkları: görüntüler üzerinde bootstrap (10 000 örnek, %95).',
             'Eşli testler: aynı görüntülerde iki modelin Dice farkı; Wilcoxon işaretli sıra testi (iki yönlü, çoklu',
             'karşılaştırma düzeltmesi yok). Bootstrap ortalamayı, Wilcoxon sıra kaymasını sınar; sonuçları farklı olabilir.',
             '"Görülmeyen 3 set" ve "Beş set" sütunları görüntü ağırlıklı ortalamadır (tablolardaki set ortalamalarının',
             'ortalaması değil). Tek eğitim koşusu: aralıklar yalnızca test örneklemesini yansıtır, eğitim seed\'ini değil.', '']

    # doğrulama: görüntü başına ortalama = tablo değeri
    bad = []
    for r in runs:
        for ds in SETS:
            d, _ = vec(r, [ds])
            t = table_mean(r, ds)
            if t is None or abs(d.mean() - t) > 6e-4:
                bad.append(f'{r}/{ds}: {d.mean():.4f} ↔ tablo {t}')
    lines += ['Doğrulama (görüntü başına ortalama = results_pranet.md): ' + ('tutarlı' if not bad else '; '.join(bad)), '']

    lines += ['## mDice ve %95 güven aralığı', '', '| Model | ' + ' | '.join(SHORT[s] for s in SETS) + ' | Görülmeyen 3 set |',
              '|---|' + '---|' * (len(SETS) + 1)]
    for lbl, r in ABL + [('Res2Net-50 (Proposed)', 'glpoly_res2net_dn')]:
        cells = []
        for ds in SETS + [None]:
            d, _ = vec(r, [ds] if ds else UNSEEN)
            cells.append(fmt_ci(d.mean(), *boot_ci(d)))
        lines.append(f'| {lbl} (`{r}`) | ' + ' | '.join(cells) + ' |')

    lines += ['', '## Eşli karşılaştırmalar (Δ = A − B, ortalama Dice farkı [%95 GA], Wilcoxon p, A daha iyi / B daha iyi görüntü)', '']
    pairs = [('Proposed', 'Baseline'), ('Proposed', '+PAM'), ('Proposed', '+HFAM'), ('+PAM', 'Baseline'),
             ('+HFAM', 'Baseline')]
    name2run = dict(ABL)
    lines += ['| A − B | ' + ' | '.join(SHORT[s] for s in SETS) + ' | Görülmeyen 3 set | Beş set |',
              '|---|' + '---|' * (len(SETS) + 2)]
    for a, b in pairs:
        cells = []
        for sets in [[s] for s in SETS] + [UNSEEN, SETS]:
            da, _ = vec(name2run[a], sets)
            db, _ = vec(name2run[b], sets)
            diff = da - db
            lo, hi = boot_ci(diff)
            p = wilcoxon(da, db).pvalue if np.any(diff != 0) else 1.0
            cells.append(f'{diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], p={p:.2g}, {int((diff > 0).sum())}/{int((diff < 0).sum())}')
        lines.append(f'| {a} − {b} | ' + ' | '.join(cells) + ' |')

    # polip boyutuna göre (görülmeyen setler birlikte; GT alanı oranının üçte birlik dilimleri)
    lines += ['', '## Polip boyutuna göre kazanç (görülmeyen üç set birlikte)', '',
              'GT alanının görüntüye oranı üç eşit sayılı dilime ayrıldı. Hücre: ortalama Dice farkı [%95 GA].', '']
    _, area = vec(ABL[-1][1], UNSEEN)
    edges = np.quantile(area, [1 / 3, 2 / 3])
    bins = np.digitize(area, edges)
    names = [f'küçük (<{edges[0]:.3f})', 'orta', f'büyük (≥{edges[1]:.3f})']
    lines += ['| A − B | ' + ' | '.join(f'{n} (n={int((bins == i).sum())})' for i, n in enumerate(names)) + ' |',
              '|---|' + '---|' * 3]
    for a, b in pairs:
        da, _ = vec(name2run[a], UNSEEN)
        db, _ = vec(name2run[b], UNSEEN)
        cells = []
        for i in range(3):
            diff = (da - db)[bins == i]
            cells.append(fmt_ci(diff.mean(), *boot_ci(diff)))
        lines.append(f'| {a} − {b} | ' + ' | '.join(cells) + ' |')

    # ek seed'ler (varsa): set ortalamalarının seed'ler üzerinde ortalama ± std
    seeded = []
    for lbl, r in ABL:
        rs = [r] + [f'{r}_s{s}' for s in (1, 2) if os.path.exists(os.path.join(RUNS, f'{r}_s{s}', 'results_pranet.md'))]
        if len(rs) > 1:
            vals = np.array([[vec(x, [ds])[0].mean() for ds in SETS] for x in rs])   # yuvarlanmamış set ortalamaları
            seeded.append((lbl, len(rs), vals))
    if seeded:
        lines += ['', '## Seed\'ler üzerinde ortalama ± std (set ortalaması mDice)', '',
                  '| Model | n | ' + ' | '.join(SHORT[s] for s in SETS) + ' | Ort. |', '|---|---|' + '---|' * (len(SETS) + 1)]
        for lbl, n, vals in seeded:
            cols = [f'{vals[:, j].mean():.3f} ± {vals[:, j].std(ddof=1):.3f}' for j in range(len(SETS))]
            avg = vals.mean(1)
            lines.append(f'| {lbl} | {n} | ' + ' | '.join(cols) + f' | {avg.mean():.3f} ± {avg.std(ddof=1):.3f} |')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'yazıldı: {OUT}')


if __name__ == '__main__':
    main()
