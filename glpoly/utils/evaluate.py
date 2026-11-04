"""Test metrikleri (BB-Res2Net-rev-3-convnext/predict_score.py'den uyarlandı, UACANet Eval).

Tahminler 0-255 PNG'dir. Sürekli haritalarda mDice/mIoU/mEφ 256 eşik üzerinden ortalanır (UACANet).
İkili maskelerde (test.py'deki preds/ ana protokolü) doğrudan maskenin Dice/IoU/Eφ değeri kullanılır.
Not: Eski kod ikili maskeleri de 256 eşikle ölçtüğü için eşik 0 değeri mDice'ı ~1/256 oranında düşürüyordu.
"""
import os

import numpy as np
from PIL import Image
from tabulate import tabulate

from .data import IMG_EXT
from .eval_functions import EnhancedMeasure, Fmeasure_calu, StructureMeasure, original_WFb

HEADERS = ['meanDic', 'meanIoU', 'wFm', 'Sm', 'meanEm', 'mae', 'maxEm', 'maxDic', 'maxIoU',
           'meanSen', 'maxSen', 'meanSpe', 'maxSpe']


def evaluate_dataset(pred_root, gt_root, legacy=False):
    thresholds = np.linspace(1, 0, 256)
    # Drive'dan gelen desktop.ini gibi dosyalar eşleşmeyi kaydırmasın
    preds = sorted(f for f in os.listdir(pred_root) if f.lower().endswith(IMG_EXT))
    gts = sorted(f for f in os.listdir(gt_root) if f.lower().endswith(IMG_EXT))
    assert len(preds) == len(gts), (pred_root, len(preds), len(gts))
    n, t = len(preds), len(thresholds)
    dice, iou, em, sen, spe = (np.zeros((n, t)) for _ in range(5))
    sm, wfm, mae = np.zeros(n), np.zeros(n), np.zeros(n)

    for i, (p, g) in enumerate(zip(preds, gts)):
        assert os.path.splitext(p)[0] == os.path.splitext(g)[0], (p, g)
        pred = np.array(Image.open(os.path.join(pred_root, p)))
        gt = np.array(Image.open(os.path.join(gt_root, g)))
        pred = pred[..., 0] if pred.ndim == 3 else pred
        gt = gt[..., 0] if gt.ndim == 3 else gt
        assert pred.shape == gt.shape, (p, pred.shape, gt.shape)
        gt = (gt.astype(np.float64) / 255 > 0.5).astype(np.float64)
        pred = pred.astype(np.float64) / 255

        sm[i] = StructureMeasure(pred, gt)
        wfm[i] = original_WFb(pred, gt)
        mae[i] = np.mean(np.abs(gt - pred))
        if not legacy and np.isin(pred, (0.0, 1.0)).all():
            # İkili maske: tarama anlamsız; eşik 0'da her piksel pozitif sayılıp ortalamayı bozmasın
            _, s, sp, d, _, io = Fmeasure_calu(pred, gt, 0.5)
            sen[i], spe[i], dice[i], iou[i] = s, sp, d, io
            em[i] = EnhancedMeasure(pred, gt)
            continue
        for j, th in enumerate(thresholds):
            _, sen[i, j], spe[i, j], dice[i, j], _, iou[i, j] = Fmeasure_calu(pred, gt, th)
            em[i, j] = EnhancedMeasure((pred >= th).astype(np.float64), gt)

    col = lambda a: a.mean(axis=0)
    return {
        'meanDic': col(dice).mean(), 'meanIoU': col(iou).mean(), 'wFm': wfm.mean(), 'Sm': sm.mean(),
        'meanEm': col(em).mean(), 'mae': mae.mean(), 'maxEm': col(em).max(), 'maxDic': col(dice).max(),
        'maxIoU': col(iou).max(), 'meanSen': col(sen).mean(), 'maxSen': col(sen).max(),
        'meanSpe': col(spe).mean(), 'maxSpe': col(spe).max(),
    }


def evaluate(pred_dir, test_root, datasets, out_dir, method, tag='', legacy=False):
    """tag verilirse çıktılar result_<tag>_<set>.csv ve results_<tag>.md adıyla yazılır.

    legacy=True: ikili maskeleri de eski kod gibi 256 eşikle ölçer (R0 ön kontrolünde eski sayılarla birebir karşılaştırma için).
    """
    sfx = f'_{tag}' if tag else ''
    rows = []
    for ds in datasets:
        r = evaluate_dataset(os.path.join(pred_dir, ds), os.path.join(test_root, ds, 'masks'), legacy)
        rows.append([ds, *[r[h] for h in HEADERS]])
        with open(os.path.join(out_dir, f'result{sfx}_{ds}.csv'), 'w') as f:
            f.write(','.join(['method', *HEADERS]) + '\n')
            f.write(method + ',' + ','.join(f'{r[h]:.4f}' for h in HEADERS) + '\n')
    tab = tabulate(rows, headers=['dataset', *HEADERS], floatfmt='.3f')
    with open(os.path.join(out_dir, f'results{sfx}.md'), 'w') as f:
        f.write(f'# {method}\n\n```\n{tab}\n```\n')
    return tab
