"""Test setlerinde tahmin üretir (eşikli ikili maske + PraNet tipi sürekli harita) ve metrikleri hesaplar.

Örnek:
    python test.py --name glpoly_res2net
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from lib.glpoly import glpoly_from_args
from utils.data import PolypEval, list_pairs
from utils.evaluate import evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--ckpt', default='best.pth')
    p.add_argument('--test_root', default=os.path.join(ROOT, 'dataset/TestDataset'))
    p.add_argument('--datasets', nargs='+', default=DATASETS)
    p.add_argument('--no_eval', action='store_true')
    p.add_argument('--only_th05', action='store_true',
                   help='yalnızca sabit 0.5 eşikli protokol (önceki koşulara sonradan eklemek için)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else
                   ('mps' if torch.backends.mps.is_available() else 'cpu'))
    opt = p.parse_args()

    run_dir = os.path.join(ROOT, 'runs', opt.name)
    ck = torch.load(os.path.join(run_dir, opt.ckpt), map_location='cpu', weights_only=False)
    a = ck['args']
    model = glpoly_from_args(a)
    model.load_state_dict(ck['model'])
    model.to(opt.device).eval()
    print(f"checkpoint epoch {ck['epoch']}, val dice {ck['best_dice']:.4f}, eşik {ck['threshold']:.2f}")

    # İki protokol:
    #  preds/        doğrulamada seçilen eşikle ikili maske (makale §4.3; eski kodun test_bb.py'si böyle)
    #  preds_pranet/ min-max normalize sürekli harita (PraNet/UACANet; literatür tablolarıyla karşılaştırma)
    #  preds_meganet/ min-max → 0.5 eşik, 256 eşikle ölçülür (MEGANet test.py + predict_score; MEGANet ile birebir)
    #  preds_th05/   sabit 0.5 eşikle ikili maske (eşik seçiminden bağımsız sonuç)
    th = ck['threshold']
    bin_dir, cont_dir = os.path.join(run_dir, 'preds'), os.path.join(run_dir, 'preds_pranet')
    meg_dir, th05_dir = os.path.join(run_dir, 'preds_meganet'), os.path.join(run_dir, 'preds_th05')
    for ds in opt.datasets:
        for d in ((th05_dir,) if opt.only_th05 else (bin_dir, cont_dir, meg_dir, th05_dir)):
            os.makedirs(os.path.join(d, ds), exist_ok=True)
        loader = DataLoader(PolypEval(list_pairs(os.path.join(opt.test_root, ds)), a['trainsize']),
                            batch_size=1, num_workers=2)
        with torch.no_grad():
            for img, gt, name in loader:
                masks, _ = model(img.to(opt.device))
                logit = F.interpolate(masks[0], size=gt.shape[-2:], mode='bilinear',
                                      align_corners=False)[0, 0].float().cpu().numpy()
                prob = 1 / (1 + np.exp(-logit))
                Image.fromarray((prob >= .5).astype(np.uint8) * 255).save(os.path.join(th05_dir, ds, name[0]))
                if opt.only_th05:
                    continue
                binary = (prob >= th).astype(np.uint8) * 255
                Image.fromarray(binary).save(os.path.join(bin_dir, ds, name[0]))
                norm = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)          # PraNet: sigmoid sonra min-max
                Image.fromarray((norm * 255).round().astype(np.uint8)).save(os.path.join(cont_dir, ds, name[0]))
                lnorm = (logit - logit.min()) / (logit.max() - logit.min() + 1e-8)     # MEGANet test.py: ham logit
                Image.fromarray(((lnorm > .5) * 255).astype(np.uint8)).save(os.path.join(meg_dir, ds, name[0]))
        print(f'{ds}: {len(loader)} tahmin')

    # metrikler CPU'da uzun sürüyor; modeli bırak ki kuyruktaki sonraki eğitimin GPU belleğini işgal etmesin
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not opt.no_eval:
        print('## sabit 0.5 eşikle ikili')
        print(evaluate(th05_dir, opt.test_root, opt.datasets, run_dir, opt.name, tag='th05'))
        if opt.only_th05:
            return
        print(f'## eşik {th:.2f} ile ikili (ana tablo)')
        print(evaluate(bin_dir, opt.test_root, opt.datasets, run_dir, opt.name))
        print('## PraNet protokolü')
        print(evaluate(cont_dir, opt.test_root, opt.datasets, run_dir, opt.name, tag='pranet'))
        print('## MEGANet protokolü')
        print(evaluate(meg_dir, opt.test_root, opt.datasets, run_dir, opt.name, tag='meganet', legacy=True))


if __name__ == '__main__':
    main()
