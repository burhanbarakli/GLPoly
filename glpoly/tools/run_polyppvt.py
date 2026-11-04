"""E1 (adil karşılaştırma): resmi Polyp-PVT'yi BİZİM protokolümüzle eğitir ve checkpoint seçiminin etkisini ölçer.

Resmi kod (../Polyp-PVT, DengPingFan/Polyp-PVT@cd71fbf) değiştirilmeden içe aktarılır: model (lib/pvt.py) ve veri
yükleyici (utils/dataloader.py). Eğitim döngüsü resmi Train.py'nin birebir kopyasıdır (AdamW lr 1e-4 wd 1e-4,
batch 16, 352 px, {0.75, 1, 1.25} ölçek, değer kırpma 0.5, structure loss P1 + P2, artırma yok [resmi varsayılan],
epoch 1..99, adjust_lr(…, 200) → sabit LR). Tek fark veride ve seçimde:
  * eğitim yalnızca bizim 1305 görüntülük eğitim bölümümüzle (runs/glpoly_convnext/split.json), doğrulama 145 görüntü;
  * resmi kod en iyi epoch'u test setinde (TestDataset/test, 798 görüntü) seçer — burada her epoch kaydedilir ve
    hem doğrulama hem test skorları günlüğe yazılır; böylece (a) doğrulamayla seçilen, (b) resmi kural gibi testle
    seçilen ve (c) son epoch karşılaştırılır.
Her epoch: doğrulama ve 5 test setinde PraNet protokolü görüntü başına Dice (sigmoid(P1+P2) → min-max, 256 eşik
ortalaması; glpoly/tools/stats.py ile aynı formül) ve resmi Train.py'deki test Dice'ı (yumuşak Dice).

    python tools/run_polyppvt.py train     # GPU; runs/polyppvt_e1/
    python tools/run_polyppvt.py final     # seçilen üç checkpoint için harita + tam ölçüm (bizim evaluate)
"""
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np

GLPOLY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(GLPOLY)
REPO = os.path.join(os.path.dirname(ROOT), 'Polyp-PVT')
RUN = os.path.join(ROOT, 'runs', 'polyppvt_e1')
TRAIN, TEST = os.path.join(ROOT, 'dataset', 'TrainDataset'), os.path.join(ROOT, 'dataset', 'TestDataset')
SETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
EPOCHS = 100            # resmi: range(1, epoch) → 99 epoch
THR = np.linspace(1, 0, 256)
GE = (np.arange(256)[:, None].astype(np.float64) / 255) >= THR[None, :]


def log(msg):
    line = f'{datetime.now():%Y-%m-%d %H:%M:%S} {msg}'
    print(line, flush=True)
    with open(os.path.join(RUN, 'log.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def pranet_dice(pred_u8, gt):
    """Bir görüntünün 256 eşik ortalaması Dice'ı (evaluate.py / stats.py ile aynı kural)."""
    p = np.bincount(pred_u8.ravel(), minlength=256) @ GE
    tp = np.bincount(pred_u8[gt].ravel(), minlength=256) @ GE
    return float(np.where(tp > 0, 2 * tp / np.maximum(gt.sum() + p, 1), 0.0).mean())


def subset_dirs():
    """split.json'daki eğitim/doğrulama görüntüleri için sabit bağlantılı (hard link) klasörler; veri kopyalanmaz."""
    yol = os.path.join(ROOT, 'runs', 'glpoly_convnext', 'split.json')
    if not os.path.exists(yol):     # açık depo: aynı liste data/split.json'da
        yol = os.path.join(ROOT, 'data', 'split.json')
    with open(yol, encoding='utf-8') as f:
        split = json.load(f)
    out = {}
    for part in ('train', 'val'):
        base = os.path.join(RUN, 'data', part)
        for sub in ('images', 'masks'):
            d = os.path.join(base, sub)
            os.makedirs(d, exist_ok=True)
            for name in split[part]:
                dst = os.path.join(d, name)
                if not os.path.exists(dst):
                    os.link(os.path.join(TRAIN, sub, name), dst)
        assert len(os.listdir(os.path.join(base, 'images'))) == len(split[part]), part
        out[part] = base
    return out, split


def import_official():
    """Resmi lib/ ve utils/ (ad alanı paketleri) bizim glpoly paketleriyle çakışmadan yüklensin: yalnızca REPO yolda."""
    sys.path[:] = [REPO] + [p for p in sys.path if os.path.abspath(p or '.') not in (GLPOLY, ROOT)]
    os.chdir(REPO)                       # PolypPVT() './pretrained_pth/pvt_v2_b2.pth' okur
    from lib.pvt import PolypPVT
    from utils.dataloader import get_loader, test_dataset
    return PolypPVT, get_loader, test_dataset


# ---- resmi Train.py / utils.py'den birebir (utils.py thop içe aktardığı için kopyalandı) ----
def structure_loss(pred, mask):
    import torch
    import torch.nn.functional as F
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
# ---------------------------------------------------------------------------------------------


def predict(model, test_dataset, root, size=352):
    """Resmi test() ile aynı ön işleme ve çıktı: sigmoid(upsample(P1 + P2)) → min-max. {ad: (uint8 harita, gt bool,
    resmi yumuşak Dice)}."""
    import torch
    import torch.nn.functional as F
    gt_root = os.path.join(root, 'masks') + '/'
    loader = test_dataset(os.path.join(root, 'images') + '/', gt_root, size)
    out = {}
    model.eval()
    with torch.no_grad():
        for _ in range(loader.size):
            image, gt, name = loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            res, res1 = model(image.cuda())
            res = F.interpolate(res + res1, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            soft = (2 * (res * gt).sum() + 1) / (res.sum() + gt.sum() + 1)          # resmi test() Dice'ı
            out[os.path.splitext(os.path.basename(name))[0]] = (np.round(res * 255).astype(np.uint8), gt > 0.5,
                                                                float(f'{soft:.4f}'))
    return out


def score(model, test_dataset, val_root):
    r = {'val': predict(model, test_dataset, val_root)}
    for ds in SETS:
        r[ds] = predict(model, test_dataset, os.path.join(TEST, ds))
    s = {k: float(np.mean([pranet_dice(m, g) for m, g, _ in v.values()])) for k, v in r.items()}
    allsoft = [x[2] for ds in SETS for x in r[ds].values()]
    s['official_test'] = float(np.mean(allsoft))           # resmi seçim ölçütü (TestDataset/test = 798 görüntü)
    s['mean5'] = float(np.mean([s[ds] for ds in SETS]))
    return s


def train():
    import torch
    import torch.nn.functional as F
    os.makedirs(RUN, exist_ok=True)
    dirs, split = subset_dirs()
    PolypPVT, get_loader, test_dataset = import_official()
    if os.path.exists(os.path.join(RUN, 'epochs.jsonl')):
        raise SystemExit(f'{RUN}/epochs.jsonl zaten var: yeniden başlatma günlüğü bozar (devam etme özelliği yok)')
    torch.manual_seed(42), np.random.seed(42), random.seed(42)

    # bellek: 448 px × batch 16 fp32 sığmazsa (Windows paylaşımlı belleğe taşar) batch 8 — tarif sapması günlüğe
    model = PolypPVT().cuda()
    batch = 16
    try:
        torch.cuda.reset_peak_memory_stats()
        for s in (256, 352, 448):
            p1, p2 = model(torch.randn(16, 3, s, s, device='cuda'))
            (p1.mean() + p2.mean()).backward()
            model.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_reserved() / 2 ** 30
    except torch.cuda.OutOfMemoryError:
        peak = float('inf')
    if peak > 14.0:
        batch = 8
    del model
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    model = PolypPVT().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4, weight_decay=1e-4)
    train_loader = get_loader(dirs['train'] + '/images/', dirs['train'] + '/masks/', batchsize=batch, trainsize=352,
                              augmentation=False)
    total_step = len(train_loader)
    log(f'Polyp-PVT (resmi kod cd71fbf), eğitim {len(split["train"])}, doğrulama {len(split["val"])}, batch {batch} '
        f'(448 px × 16 ayrılan bellek {peak:.1f} GB){"; TARİF SAPMASI: batch 8" if batch != 16 else ""}, '
        f'{total_step} adım/epoch, epoch 1..{EPOCHS - 1}')
    for epoch in range(1, EPOCHS):
        adjust_lr(optimizer, 1e-4, epoch, 0.1, 200)
        model.train()
        t0, losses = time.time(), []
        for i, (images, gts) in enumerate(train_loader, start=1):
            for rate in (0.75, 1, 1.25):
                optimizer.zero_grad()
                images_, gts_ = images.cuda(), gts.cuda()
                trainsize = int(round(352 * rate / 32) * 32)
                if rate != 1:
                    images_ = F.interpolate(images_, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts_ = F.interpolate(gts_, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                P1, P2 = model(images_)
                loss = structure_loss(P1, gts_) + structure_loss(P2, gts_)
                loss.backward()
                clip_gradient(optimizer, 0.5)
                optimizer.step()
                if rate == 1:
                    losses.append(loss.item())
        torch.save(model.state_dict(), os.path.join(RUN, f'epoch_{epoch:03d}.pth'))
        t1 = time.time()
        s = score(model, test_dataset, dirs['val'])
        with open(os.path.join(RUN, 'epochs.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps({'epoch': epoch, 'loss': float(np.mean(losses)), **s}) + '\n')
        log(f'EPOCH {epoch:03d} loss {np.mean(losses):.4f} val {s["val"]:.4f} ort5 {s["mean5"]:.4f} '
            f'resmi_test {s["official_test"]:.4f} ' + ' '.join(f'{d} {s[d]:.3f}' for d in SETS) +
            f' eğitim {(t1 - t0) / 60:.1f} dk ölçüm {(time.time() - t1) / 60:.1f} dk')
    log('EĞİTİM BİTTİ')


def final():
    """Üç seçim kuralının checkpoint'leri için harita yaz + bizim evaluate ile tam ölçüm; seçim etkisi özeti."""
    rows = [json.loads(ln) for ln in open(os.path.join(RUN, 'epochs.jsonl'), encoding='utf-8')]
    pick = {'val': max(rows, key=lambda r: r['val'])['epoch'],
            'official_test': max(rows, key=lambda r: r['official_test'])['epoch'],
            'last': rows[-1]['epoch']}
    dirs, _ = subset_dirs()
    import torch
    PolypPVT, _, test_dataset = import_official()
    for tag, ep in pick.items():
        model = PolypPVT().cuda()
        model.load_state_dict(torch.load(os.path.join(RUN, f'epoch_{ep:03d}.pth'), map_location='cuda'))
        for ds in SETS:
            d = os.path.join(RUN, f'preds_{tag}', ds)
            os.makedirs(d, exist_ok=True)
            for name, (m, _, _) in predict(model, test_dataset, os.path.join(TEST, ds)).items():
                from PIL import Image
                Image.fromarray(m).save(os.path.join(d, name + '.png'))
        del model
        torch.cuda.empty_cache()
    # tam ölçüm bizim kodla, ayrı süreçte (glpoly paketleri resmi lib/utils ile çakışmasın)
    import subprocess
    for tag in pick:
        os.makedirs(os.path.join(RUN, f'eval_{tag}'), exist_ok=True)
        subprocess.check_call([sys.executable, '-c',
                               'import sys; sys.path.insert(0, sys.argv[1]); from utils.evaluate import evaluate; '
                               'print(evaluate(sys.argv[2], sys.argv[3], sys.argv[4].split(","), sys.argv[5], sys.argv[6]))',
                               GLPOLY, os.path.join(RUN, f'preds_{tag}'), TEST, ','.join(SETS),
                               os.path.join(RUN, f'eval_{tag}'), f'Polyp-PVT ({tag}, epoch {pick[tag]})'],
                              cwd=GLPOLY)
    best_mean = max(rows, key=lambda r: r['mean5'])
    lines = ['# E1: resmi Polyp-PVT, bizim protokol (1305 eğitim / 145 doğrulama)', '',
             f'Seçilen epoch\'lar: doğrulama {pick["val"]}, resmi kural (test) {pick["official_test"]}, son {pick["last"]}; '
             f'test ortalamasının en iyisi epoch {best_mean["epoch"]} ({best_mean["mean5"]:.4f}).', '',
             '| Seçim | Epoch | ' + ' | '.join(SETS) + ' | Ort. |', '|---|---|' + '---|' * (len(SETS) + 1)]
    for tag, ep in pick.items():
        r = next(x for x in rows if x['epoch'] == ep)
        lines.append(f'| {tag} | {ep} | ' + ' | '.join(f'{r[d]:.3f}' for d in SETS) + f' | {r["mean5"]:.3f} |')
    lines += ['', 'Tam ölçütler: `eval_<seçim>/results.md` (PraNet protokolü, bizim evaluate). Epoch günlüğü: `epochs.jsonl`.']
    with open(os.path.join(RUN, 'ozet.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    {'train': train, 'final': final}[sys.argv[1]]()
