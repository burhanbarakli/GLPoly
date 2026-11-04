"""Karşılaştırılan yöntemlerin yayımlanmış tahmin haritalarını bizim değerlendirme kodumuzla ölçer (adil karşılaştırma).

Haritalar `downloads/` altındaki arşivlerden (README bağlantıları, revizyon/kod_erisimi.md) okunur, gri tonlamaya
çevrilip GT ile aynı ada ve boyuta getirilerek runs/_ext/<yöntem>/preds/<set>/ altına yazılır; ardından
utils.evaluate (PraNet protokolü: sürekli haritalarda 256 eşik ortalaması, ikili haritalarda doğrudan) çalıştırılır.
Sürekli haritalar, PraNet aracındaki ve bizim preds_pranet'teki gibi harita başına min-max ile [0, 255]'e yayılır.
MEGANet yalnızca ikili harita yayımlıyor: ortalama sütunları tek eşikli değerdir (bizim ana protokolümüzle aynı tür).
Boyut farkı olursa harita GT boyutuna bilineer ölçeklenir ve sayısı rapora yazılır.

    python tools/eval_external.py [--methods PraNet SANet ...] [--jobs 3]
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import IMG_EXT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DL, TEST, OUT = os.path.join(ROOT, 'downloads'), os.path.join(ROOT, 'dataset', 'TestDataset'), os.path.join(ROOT, 'runs', '_ext')
DATASETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
# yöntem → (harita klasörü, kaynak)
SOURCES = {
    'U-Net': ('pranet_toolbox/EvaluationTool_New/map_new/UNet', 'PraNet değerlendirme paketi'),
    'U-Net++': ('pranet_toolbox/EvaluationTool_New/map_new/UNet++', 'PraNet değerlendirme paketi'),
    'SFA': ('pranet_toolbox/EvaluationTool_New/map_new/SFA', 'PraNet değerlendirme paketi'),
    'PraNet': ('pranet_toolbox/EvaluationTool_New/map_new/PraNet', 'PraNet değerlendirme paketi (resmi)'),
    'SANet': ('sanet_predictions/SANet', 'SANet deposu (resmi)'),
    'MSNet': ('msnet_results', 'MSNet-M2SNet deposu (resmi)'),
    'M2SNet': ('m2snet_results/results/image_polyp_segmentation', 'MSNet-M2SNet deposu (resmi)'),
    'MEGANet (ResNet-34)': ('meganet_resnet_maps/MEGANet-ResNet', 'MEGANet deposu (resmi; ikili haritalar)'),
    'DCRNet': ('polyp_pvt_baselines/compared_results/DCRNet', 'Polyp-PVT karşılaştırma arşivi'),
    'ACSNet': ('polyp_pvt_baselines/compared_results/ACSNet', 'Polyp-PVT karşılaştırma arşivi'),
    'EU-Net': ('polyp_pvt_baselines/compared_results/EU-Net', 'Polyp-PVT karşılaştırma arşivi'),
    'HarDNet-MSEG': ('polyp_pvt_baselines/compared_results/MSEG', 'Polyp-PVT karşılaştırma arşivi'),
    'CaraNet': ('caranet_maps/map/Caranet', 'CaraNet deposu (resmi)'),
    'CFA-Net': ('cfanet_maps/CFANet', 'CFA-Net deposu (resmi)'),
    'Polyp-PVT': ('polyp_pvt_maps/result_map/PolypPVT', 'Polyp-PVT deposu (resmi)'),
    # haritası yayımlanmamış, resmi ağırlıkla bizim ürettiğimiz (tools/infer_external.py)
    'PraNet-V2 (Res2Net-50)': ('maps_pranetv2_res', 'PraNet-V2 resmi ağırlığı RES-V2.pth, infer_external.py'),
    'PraNet-V2 (PVTv2-B2)': ('maps_pranetv2_pvt', 'PraNet-V2 resmi ağırlığı PVT-V2.pth, infer_external.py'),
    'SSFormer-L': ('maps_ssformer_l', 'SSFormer resmi ağırlığı ssformer_L.pth, infer_external.py'),
    # resmi kodla BİZİM protokolümüzle yeniden eğitilen (tools/run_retrain.py; doğrulamayla seçilen epoch / son epoch)
    **{f'{lab} (retrain{s})': (f'maps_{m}_retrain{d}', f'run_retrain.py, {"son epoch" if d else "doğrulamayla seçilen epoch"}')
       for m, lab in (('meganet_r34', 'MEGANet (ResNet-34)'), ('hsnet', 'HSNet'), ('cafenet', 'CAFE-Net'), ('pranetv2_res', 'PraNet-V2 (Res2Net-50)'),
                      ('pranetv2_pvt', 'PraNet-V2 (PVTv2-B2)'), ('ssformer_l', 'SSFormer-L'), ('fcbformer', 'FCBFormer'),
                      ('baasnet', 'BAASNet'), ('hardmseg', 'HarDNet-MSEG'), ('futranshnet', 'Fu-TransHNet'))
       for s, d in (('', ''), (', son', '_last'))},
}


def slug(m):
    # '+' ayrı tutulur: U-Net ile U-Net++ aynı klasöre düşmesin (U_Net / U_Netpp)
    return ''.join(c if c.isalnum() else 'p' if c == '+' else '_' for c in m).strip('_')


def set_dir(base, ds):
    for name in (ds, ds.replace('-', '_')):
        p = os.path.join(DL, base, name)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f'{base}/{ds}')


def prepare(method):
    """Haritaları GT adlarına/boyutlarına getirir; (ölçeklenen sayısı, ikili mi) döndürür."""
    base, _ = SOURCES[method]
    resized, binary = 0, True
    for ds in DATASETS:
        src, dst = set_dir(base, ds), os.path.join(OUT, slug(method), 'preds', ds)
        os.makedirs(dst, exist_ok=True)
        maps = {os.path.splitext(f)[0]: f for f in os.listdir(src) if f.lower().endswith(IMG_EXT)}
        gdir = os.path.join(TEST, ds, 'masks')
        for g in sorted(f for f in os.listdir(gdir) if f.lower().endswith(IMG_EXT)):
            stem = os.path.splitext(g)[0]
            if stem not in maps:
                raise FileNotFoundError(f'{method}/{ds}: {stem} haritası yok')
            gt_size = Image.open(os.path.join(gdir, g)).size
            im = Image.open(os.path.join(src, maps[stem])).convert('L')
            if im.size != gt_size:
                im = im.resize(gt_size, Image.BILINEAR)
                resized += 1
            a = np.asarray(im)
            if np.isin(a, (0, 255)).all():
                pass                                    # ikili harita (MEGANet): olduğu gibi
            else:
                binary = False
                lo, hi = float(a.min()), float(a.max())  # PraNet aracı (main.m: mapminmax) ve bizim preds_pranet gibi
                a = np.round((a - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
            Image.fromarray(a).save(os.path.join(dst, stem + '.png'))
    return resized, binary


def run_one(method):
    from utils.evaluate import evaluate
    resized, binary = prepare(method)
    out = os.path.join(OUT, slug(method))
    print(evaluate(os.path.join(out, 'preds'), TEST, DATASETS, out, method))
    with open(os.path.join(out, 'kaynak.txt'), 'w', encoding='utf-8') as f:
        f.write(f'{method}\nkaynak: {SOURCES[method][1]} ({SOURCES[method][0]})\n'
                f'GT boyutuna ölçeklenen harita: {resized}\nikili harita: {binary}\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='+', default=list(SOURCES))
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--one', default='')
    opt = p.parse_args()
    if opt.one:
        run_one(opt.one)
        return
    os.makedirs(OUT, exist_ok=True)
    todo = [m for m in opt.methods if not os.path.exists(os.path.join(OUT, slug(m), 'results.md'))]
    procs = []
    for m in todo:
        while sum(q.poll() is None for q in procs) >= opt.jobs:
            time.sleep(5)
        log = open(os.path.join(OUT, f'{slug(m)}_stdout.txt'), 'w', encoding='utf-8')
        procs.append(subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__), '--one', m],
                                      cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      stdout=log, stderr=subprocess.STDOUT))
        print(f'başladı: {m}', flush=True)
    for q in procs:
        q.wait()
    for m in opt.methods:
        r = os.path.join(OUT, slug(m), 'results.md')
        print(f'{m}: {"tamam" if os.path.exists(r) else "HATA (bkz. " + slug(m) + "_stdout.txt)"}')


if __name__ == '__main__':
    main()
