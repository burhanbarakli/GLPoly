"""Adil karşılaştırma: karşılaştırılan yöntemleri resmi kodlarıyla BİZİM protokolümüzle yeniden eğitir.

Şablon tools/run_polyppvt.py (E1). Depoların (downloads/repos/, güvenilmeyen kod) hazır betikleri çalıştırılmaz;
yalnızca model / kayıp / veri yükleyici sınıfları değiştirilmeden içe aktarılır. Resmi eğitim döngüsü (optimizer, LR
takvimi, batch, ölçekler, artırmalar, kayıp, gradyan kırpma) birebir kopyalanır; yalnızca veri ve checkpoint seçimi
farklıdır:
  * eğitim yalnızca bizim 1305 görüntülük eğitim bölümümüz, doğrulama 145 görüntü (runs/glpoly_convnext/split.json);
  * her epoch yalnızca doğrulamada PraNet protokolü Dice (256 eşik ortalaması; run_polyppvt.py / stats.py ile aynı
    formül) ölçülür — 5 test seti epoch başına ölçülmez (E1'den fark: çok yavaş). Checkpoint = en iyi doğrulama
    epoch'u (best.pth); son epoch da tutulur (last.pth, devam etme bilgisiyle).
  * `final`: best.pth ve last.pth için 5 test setinde olasılık haritası (<stem>.png, 0–255 gri, GT boyutunda, harita
    başına min-max) → downloads/maps_<yöntem>_retrain/<set>/ ve downloads/maps_<yöntem>_retrain_last/<set>/
    (eval_external.py prepare() düzeni; SOURCES girdileri elle eklenir).

Ön-eğitimli omurgalar yalnızca yerel dosyalardan yüklenir: kurucuların sabit/göreli yollu torch.load çağrıları
(./pretrained_pth/pvt_v2_b2.pth, ./models/pvt_v2_b2.pth, ../models/res2net50_v1b_26w_4s-3cf99910.pth,
/mnt/.../mit_b4.pth) yerel dosyaya yönlendirilir; model_zoo / hub indirmeleri kapatılır (internete çıkılmaz).
Tarif sapmaları RECIPES içinde ve günlükte 'TARİF SAPMASI' olarak işaretlidir.

    python tools/run_retrain.py <yöntem> train     # GPU; runs/retrain_<yöntem>/ (kesilirse last.pth'den devam eder)
    python tools/run_retrain.py <yöntem> final     # best + last için 5 test setine harita
    yöntem: hsnet | cafenet | pranetv2_res | pranetv2_pvt | ssformer_l | meganet_r34 | fcbformer | baasnet | hardmseg |
            futranshnet (önce hardmseg: Fu-TransHNet onun bizim bölmeyle eğitilmiş best.pth dosyasından başlar)
meganet_r34 resmi kodu downloads/repos/ yerine ../MEGANet/MEGANet-ResNet'ten (RECIPES 'repo_dir') içe aktarılır.
Yalnızca deneme (smoke) için: --run-dir, --device cpu, --subset N, --max-steps N, --epochs N, --stop-after E,
--workers N, --threads N, --no-pretrain.
"""
import argparse
import contextlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import types
from datetime import datetime

import numpy as np

sys.dont_write_bytecode = True      # downloads/repos/ altına __pycache__ yazılmasın (depolar değiştirilmez)
GLPOLY =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(GLPOLY)
DL = os.path.join(ROOT, 'downloads')
REPOS = os.path.join(DL, 'repos')
TRAIN, TEST = os.path.join(ROOT, 'dataset', 'TrainDataset'), os.path.join(ROOT, 'dataset', 'TestDataset')
SPLIT = os.path.join(ROOT, 'runs', 'glpoly_convnext', 'split.json')
if not os.path.exists(SPLIT):     # açık depo: aynı liste data/split.json'da
    SPLIT = os.path.join(ROOT, 'data', 'split.json')
SETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
IMG_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')
THR = np.linspace(1, 0, 256)
GE = (np.arange(256)[:, None].astype(np.float64) / 255) >= THR[None, :]
MEM_LIMIT_GB = 14.0     # 16 GB kartta bunun üstü Windows'ta paylaşımlı belleğe taşar (E1 ile aynı kural)

# yerel ön-eğitim dosyaları: kurucuların torch.load yolunun dosya adı → yerel dosya
PRETRAIN = {
    'pvt_v2_b2.pth': os.path.join(os.path.dirname(ROOT), 'Polyp-PVT', 'pretrained_pth', 'pvt_v2_b2.pth'),
    'res2net50_v1b_26w_4s-3cf99910.pth': os.path.join(ROOT, 'pretrained', 'res2net50_v1b_26w_4s-3cf99910.pth'),
    'mit_b4.pth': os.path.join(DL, 'w_mit_b4', 'mit_b4.pth'),
    # MEGANet: torchvision models.resnet34(pretrained=True) dosyasının aynısı (torch hub önbelleği; indirme yok)
    'resnet34-b627a593.pth': os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints',
                                          'resnet34-b627a593.pth'),
    'pvt_v2_b3.pth': os.path.join(DL, 'w_fcbformer', 'pvt_v2_b3.pth'),
    'resnet101-cd907fc2.pth': os.path.join(DL, 'w_baasnet', 'resnet101-cd907fc2.pth'),
    'hardnet68.pth': os.path.join(DL, 'w_futranshnet', 'hardnet68.pth'),
    'deit_small_patch16_224_cd65a155 .pth': os.path.join(DL, 'w_futranshnet', 'deit_small_patch16_224-cd65a155.pth'),
    'HarD-MSEG-best.pth': os.path.join(ROOT, 'runs', 'retrain_hardmseg', 'best.pth'),   # bizim bölmeyle eğitilmiş
}
# bu dosyalar kurucuda her zaman yüklenir (resmi kurucuda koşulsuz; yoksa strict yükleme boş sözlükte hata verir)
ALWAYS_LOAD = {'pvt_v2_b3.pth', 'hardnet68.pth'}

# ---- yöntem başına resmi tarif (kaynak: depodaki eğitim betiği; sapmalar 'sapma' alanında) ----------------------
# epochs = resmi döngü aralığı (başlangıç, bitiş hariç); lr_drop = {epoch: çarpan}, epoch başında uygulanır.
RECIPES = {
    'hsnet': dict(
        repo='HSNet', src='Train.py (+ utils/dataloader.py, lib/pvt.py HSNet)', backbone='pvt_v2_b2.pth',
        optimizer='AdamW', lr=5e-5, wd=1e-4, batch=8, size=352, rates=(0.75, 1, 1.25), clip=0.5,
        epochs=(1, 100), lr_drop={15: 0.5, 30: 0.5}, workers=4, seed=42,
        aug=True,   # resmi bayrak True (bool) ama dataloader 'True' (str) ile karşılaştırıyor → fiilen artırma YOK
        loss='structure_loss(P1..P4) toplamı (wbce reduce="none" → eski API ile skaler ortalama, resmi gibi)',
        output='sigmoid(P1+P2+P3+P4)', sapma=[]),
    'cafenet': dict(
        repo='CAFE-Net', src='train.py (+ utils/dataloader.py, lib/pvt.py CAFE)', backbone='pvt_v2_b2.pth',
        optimizer='AdamW', lr=1e-4, wd=1e-4, batch=16, size=352, rates=(0.75, 1, 1.25), clip=0.5,
        epochs=(1, 150), lr_drop={50: 0.5, 90: 0.5, 120: 0.5}, workers=4, seed=42,
        aug=True,   # == True → RandomRotation(90) + V/H flip + ColorJitter(0.4, 0.5, 0.25, 0.01) (yalnız görüntü)
        loss='structure_loss(5 çıktı) toplamı (wbce reduction="mean" → ağırlıksız skaler BCE, resmi gibi)',
        output='sigmoid(x_out) (test.py: P1)', sapma=[]),
    'pranetv2_res': dict(
        repo='PraNet-V2/binary_seg', src='MyTrain_med.py --model_type PraNet-V2', backbone='res2net50_v1b_26w_4s-3cf99910.pth',
        optimizer='Adam', lr=1e-4, wd=0.0, batch=8, size=352, rates=(0.75, 1, 1.25), clip=0.5,
        epochs=(1, 30), lr_drop={}, workers=4, seed=42, aug=None, num_class=1,
        # adjust_lr(opt, 1e-4, epoch, 0.1, 50): lr *= 0.1 ** (epoch // 50) = 1 → 29 epoch boyunca sabit LR
        loss='structure_loss(fg, bg, gt, 1-gt) 4 yan çıktı toplamı (wbce + wiou + 0.8 * wbce_bg)',
        output='sigmoid(p2+p3+p4+p5)',
        sapma=['cuDNN açık (resmi betik ve utils/dataloader.py torch.backends.cudnn.enabled=False yapıyor; yalnız hız)',
               "resmi MyTrain_med.py veri yükleyiciden sonra sys.exit(0) ile çıkıyor (yayımlanmış hata); bu satır yok sayıldı"]),
    'pranetv2_pvt': dict(
        repo='PraNet-V2/binary_seg', src='MyTrain_med.py --model_type PVT-PraNet-V2', backbone='pvt_v2_b2.pth',
        optimizer='Adam', lr=1e-4, wd=0.0, batch=8, size=352, rates=(0.75, 1, 1.25), clip=0.5,
        epochs=(1, 30), lr_drop={}, workers=4, seed=42, aug=None, num_class=1,
        loss='structure_loss(fg, bg, gt, 1-gt) 4 yan çıktı toplamı (wbce + wiou + 0.8 * wbce_bg)',
        output='sigmoid(p2+p3+p4+p5)',
        sapma=['cuDNN açık (resmi: kapalı; yalnız hız)',
               "resmi MyTrain_med.py sys.exit(0) hatası yok sayıldı"]),
    'ssformer_l': dict(
        repo='ssformer', src='train.py + configs/ssformer-L.yaml (mit_PLD_b4)', backbone='mit_b4.pth',
        optimizer='AdamW', lr=1e-4, wd=0.01, batch=32, size=352, rates=(1,), clip=None,
        epochs=(0, 200), lr_step=40, lr_gamma=0.1, workers=8, seed=1, aug='yaml Train_transform_list',
        # resmi zamanlayıcı: GradualWarmupScheduler(multiplier=1, total_epoch=1) + StepLR(step_size=200, gamma=0.1),
        # step(epoch) ile → kapalı biçim: lr(0) = 0, lr(e≥1) = 1e-4 * 0.1 ** ((e - 1) // step)
        loss='dice_bce_loss(sigmoid(out), gt) (utils/loss.py; sigmoid iki kez uygulanıyor, resmi gibi)',
        output='sigmoid(out) → GT boyutuna',
        sapma=['epoch 2000 → 200 (makale: "The training period is 200 epochs"; config max_epoch 2000)',
               'LR adımı 200 → 40 epoch (makale: "decay rate of 0.1, and a decay period of 40 epochs"; kod StepLR(200))',
               'mmcv.cnn.ConvModule ve einops yer tutucu (mmcv 1.x ConvModule: conv(bias yok)+BN+ReLU, kaiming fan_out başlatma)']),
    'meganet_r34': dict(
        repo='MEGANet/MEGANet-ResNet', repo_dir=os.path.join(os.path.dirname(ROOT), 'MEGANet', 'MEGANet-ResNet'),
        src='train.py (+ lib/EGANet.py EGANetModel, utils/loss.py DeepSupervisionLoss, utils/dataloader.py get_loader)',
        backbone='resnet34-b627a593.pth (torchvision ImageNet, yerel)',
        optimizer='SGD', lr=1e-3, momentum=0.9, wd=1e-5, batch=16, size=352, rates=(0.75, 1, 1.25), clip=0.5,
        # resmi: LambdaLR(lambda e: 1 - (e / opt.epoch) ** 0.9), scheduler.step() epoch sonunda → epoch e'de
        # lr = 1e-3 * (1 - (e / 100) ** 0.9); poly toplamı kullanıcı kararıyla 100 (resmi 200)
        epochs=(0, 100), poly_epochs=100, power=0.9, workers=4,
        seed=42,    # resmi betik tohum ayarlamıyor; diğer yöntemlerle aynı 42
        aug=True,   # get_loader(augmentation=True) varsayılanı → RandomRotate(0–360, p=0.5) + V/H çevirme (görüntü+maske)
        loss='DeepSupervisionLoss() = BceDiceLoss(sigmoid(out_k), gt ↓2^k) k=0..4 toplamı (5 çıktının hepsi; resmi sınıf)',
        output='sigmoid(out1 = predicts[0]) → GT boyutuna (eval_meganet.py PraNet protokolü; resmi test.py: sigmoid '
               'yok, min-max, 0.5 eşikle ikili harita)',
        sapma=['epoch 200 → 100 (kullanıcı kararı; poly LR 100 epoch toplamıyla: lr = 1e-3·(1 − e/100)^0.9)',
               'bizim protokol (kullanıcı kararı): eğitim 1305 görüntü (resmi: 1450 tamamı), epoch seçimi 145 '
               'görüntülük doğrulamada Dice ile (resmi: son epoch, doğrulama yok)']),
    'fcbformer': dict(
        repo='FCBFormer', src='train.py (+ Data/dataset.py SegDataset, Metrics/losses.py, Models/models.py)',
        backbone='pvt_v2_b3.pth', optimizer='AdamW', lr=1e-4, wd=0.01, batch=16, size=352, rates=(1,), clip=None,
        epochs=(1, 201), lr_drop={}, plateau=dict(factor=0.5, patience=10, min_lr=1e-6), workers=4, seed=42, aug=True,
        loss='SoftDiceLoss(logits, gt) + BCELoss(sigmoid(logits), gt)', output='logit GT boyutuna, sonra sigmoid',
        sapma=['veri bölmesi: resmi Kvasir ve ClinicDB için ayrı rastgele 80/10/10; burada 1305/145 tek model',
               'ReduceLROnPlateau ve epoch seçimi doğrulama PraNet Dice ile (resmi: 0.5 eşikli DiceScore)',
               'DataLoader işçi sayısı 4 (resmi: CPU sayısı; yalnız hız)',
               'çıktı sürekli harita, GT boyutunda, harita başına min-max (resmi: 352 çözünürlükte ikili, logit > 0)', 'seed 42 (resmi betik tohum ayarlamıyor)']),
    'baasnet': dict(
        repo='BAASNet', src='train.py (+ dataloader.py, loss.py, model.py)', backbone='resnet101-cd907fc2.pth',
        optimizer='AdamW', lr=1e-4, wd=1e-5, batch=32, min_accum=2, size=352, rates=(1,), clip=None,
        epochs=(0, 300), lr_drop={}, plateau=dict(factor=0.5, patience=5, min_lr=0.0), workers=8, seed=42, aug=True,
        loss='CombinedLoss(dice 1, bce 1, kenar 0.5)', output='sigmoid(out); giriş 0-255, normalizasyonsuz (resmi gibi)',
        sapma=['veri bölmesi: resmi betik --train_data/--valid_data alır (makale: Kvasir 900/100 ve başka setler); '
               'burada 1305/145',
               'ReduceLROnPlateau ve epoch seçimi doğrulama PraNet Dice ile (resmi: doğrulama kaybı)',
               'giriş 352 (resmi 512): diğer yöntemlerle aynı çözünürlük, kullanıcı kararı',
               'etkin batch 32: resmi yorum "16*2 = 32 effective batch size" (argparse varsayılanlarıyla 32 × 2 = 64 '
               'olurdu; yoruma uyuldu; Dice mikro batch üzerinden)', 'çıktı sürekli harita, GT boyutunda, harita başına min-max (resmi: model çözünürlüğünde ikili, > 0.5)', 'seed 42 (resmi betik tohum ayarlamıyor)']),
    'hardmseg': dict(
        repo='HarDNet-MSEG', src='Train.py (+ lib/HarDMSEG.py, utils/dataloader.py)', backbone='hardnet68.pth',
        optimizer='Adam', lr=1e-4, wd=0.0, batch=16, size=352, rates=(0.75, 1, 1.25), clip=0.5,
        epochs=(1, 100), lr_drop={}, workers=4, seed=42, aug=False,
        loss='structure_loss(çıktı) (wbce reduce="none" → skaler ortalama, resmi gibi)', output='sigmoid(out)',
        sapma=['eğitim 1305 görüntü (resmi: 1450 tamamı)',
               'epoch seçimi doğrulamada Dice ile (resmi: Train.py her epoch best=0 yaptığından fiilen son epoch)',
               'seed 42 (resmi betik tohum ayarlamıyor)']),
    'futranshnet': dict(
        repo='Fu-TransHNet', src='train_isic.py (+ lib/TransFuse.py TransFuse_S, utils/dataloader.py SkinDataset, '
                                 'process_train.py)', backbone='HarD-MSEG-best.pth + DeiT-S + hardnet68',
        optimizer='Adam', betas=(0.5, 0.999), lr=7e-5, wd=0.0, batch=20, size=352, rates=(1,), clip=None, clip_norm=2.0,
        epochs=(1, 201), lr_drop={}, workers=4, seed=42, aug=True,
        loss='update_w ağırlıklı structure_loss (n=1, resmi gibi)',
        output='sigmoid(0.3423·map2 + 0.3269·map1 + 0.3308·map_x) (test_isic.py sabit ağırlıkları)',
        sapma=['veri bölmesi: resmi rastgele 80/20 (seed yok); burada 1305/145',
               'HarD-MSEG-best.pth bizim 1305/145 bölmeyle eğitilen HarDNet-MSEG (Fu-TransHNet deposunda yok; yayımlanan ağırlık bizim bölmeye uymuyor)',
               'epoch seçimi doğrulamada PraNet Dice ile (resmi: her 10 epoch kayıt, seçim kuralı yok)',
               'çıktı sürekli harita, GT boyutunda, harita başına min-max (resmi: 352 çözünürlükte ikili, > 0.5)', 'seed 42 (resmi betik tohum ayarlamıyor)',
               'persistent_workers (yalnız hız); maskeler float32 (resmi float64)']),
}


def log(run, msg):
    line = f'{datetime.now():%Y-%m-%d %H:%M:%S} {msg}'
    print(line, flush=True)
    with open(os.path.join(run, 'log.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def pranet_dice(pred_u8, gt):
    """Bir görüntünün 256 eşik ortalaması Dice'ı (evaluate.py / stats.py / run_polyppvt.py ile aynı kural)."""
    p = np.bincount(pred_u8.ravel(), minlength=256) @ GE
    tp = np.bincount(pred_u8[gt].ravel(), minlength=256) @ GE
    return float(np.where(tp > 0, 2 * tp / np.maximum(gt.sum() + p, 1), 0.0).mean())


def subset_dirs(run, limit=None):
    """split.json eğitim/doğrulama görüntüleri için sabit bağlantılı (hard link) klasörler (run_polyppvt.py gibi).
    limit yalnız deneme içindir (ilk N ad)."""
    with open(SPLIT, encoding='utf-8') as f:
        split = json.load(f)
    assert not set(split['train']) & set(split['val']), 'eğitim/doğrulama kesişiyor'
    if limit:
        split = {k: v[:limit] for k, v in split.items()}
    out = {}
    for part in ('train', 'val'):
        base = os.path.join(run, 'data', part)
        for sub in ('images', 'masks'):
            d = os.path.join(base, sub)
            os.makedirs(d, exist_ok=True)
            for name in split[part]:
                dst = os.path.join(d, name)
                if not os.path.exists(dst):
                    os.link(os.path.join(TRAIN, sub, name), dst)
            assert len(os.listdir(d)) == len(split[part]), (part, sub)
        out[part] = base
    return out, split


# ---- yer tutucular: kurulu olmayan, kullanılmayan içe aktarmalar --------------------------------------------------
def _data_stubs():
    """Veri modüllerinin içe aktarıp kullanmadığı paketler (CAFE-Net: skimage.measure; SSFormer: albumentations).
    Modül düzeyinde kurulur: Windows'ta DataLoader işçileri (spawn) bu dosyayı __mp_main__ olarak yeniden yükler ve
    veri kümesini ancak bundan sonra açar."""
    def unavailable(*a, **k):
        raise NotImplementedError('yer tutucu: bu işlev kullanılmamalı')
    if importlib.util.find_spec('skimage') is None and 'skimage' not in sys.modules:
        sk, skm = types.ModuleType('skimage'), types.ModuleType('skimage.measure')
        skm.label = skm.regionprops = skm.find_contours = unavailable
        sk.measure = skm
        sys.modules.update({'skimage': sk, 'skimage.measure': skm})
    if importlib.util.find_spec('albumentations') is None and 'albumentations' not in sys.modules:
        sys.modules['albumentations'] = types.ModuleType('albumentations')
    if importlib.util.find_spec('pandas') is None and 'pandas' not in sys.modules:   # BAASNet: içe aktarılıyor, kullanılmıyor
        sys.modules['pandas'] = importlib.util.module_from_spec(importlib.util.spec_from_loader('pandas', None))


_data_stubs()


def _ssformer_model_stubs():
    """SSFormer-L: mmcv.cnn.ConvModule ve einops yer tutucuları (infer_external.stub_ssformer_deps yaklaşımı). Eğitim
    için mmcv 1.x başlatması da kopyalanır: ConvModule.init_weights → kaiming_init(conv, mode='fan_out',
    nonlinearity='relu', normal, bias 0) ve constant_init(bn, 1, 0). Modül adları mmcv ile aynı (conv / bn / activate)."""
    import torch.nn as nn

    class ConvModule(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_cfg=None,
                     act_cfg=dict(type='ReLU')):
            super().__init__()
            assert norm_cfg is None or norm_cfg.get('type') == 'BN', norm_cfg
            assert act_cfg is None or act_cfg.get('type') == 'ReLU', act_cfg
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=norm_cfg is None)
            self.bn = nn.BatchNorm2d(out_channels) if norm_cfg is not None else None
            self.activate = nn.ReLU(inplace=True) if act_cfg is not None else None
            nn.init.kaiming_normal_(self.conv.weight, a=0, mode='fan_out', nonlinearity='relu')
            if self.conv.bias is not None:
                nn.init.constant_(self.conv.bias, 0)
            if self.bn is not None:
                nn.init.constant_(self.bn.weight, 1)
                nn.init.constant_(self.bn.bias, 0)

        def forward(self, x):
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            return self.activate(x) if self.activate is not None else x

    def unavailable(*a, **k):
        raise NotImplementedError('mmcv/einops yer tutucusu: bu işlev kullanılmamalı')

    cnn = types.ModuleType('mmcv.cnn')
    cnn.ConvModule, cnn.build_norm_layer, cnn.build_activation_layer = ConvModule, unavailable, unavailable
    mmcv = types.ModuleType('mmcv')
    mmcv.cnn = cnn
    ein = types.ModuleType('einops')
    ein.rearrange = unavailable
    ein_layers, ein_torch = types.ModuleType('einops.layers'), types.ModuleType('einops.layers.torch')
    ein_torch.Rearrange = unavailable
    ein.layers, ein_layers.torch = ein_layers, ein_torch
    sys.modules.update({'mmcv': mmcv, 'mmcv.cnn': cnn, 'einops': ein, 'einops.layers': ein_layers,
                        'einops.layers.torch': ein_torch})


def import_official(method):
    """Depo yolda tek başına: resmi lib/ utils/ models/ paketleri bizim glpoly paketleriyle çakışmasın."""
    repo = RECIPES[method].get('repo_dir') or os.path.join(REPOS, *RECIPES[method]['repo'].split('/'))
    sys.path[:] = [repo] + [p for p in sys.path if os.path.abspath(p or '.') not in (GLPOLY, ROOT, repo)]
    return repo


@contextlib.contextmanager
def local_pretrain(pretrained, loaded):
    """Kurucu içindeki torch.load(<sabit yol>) → yerel dosya (map_location='cpu', weights_only=True); internetten
    indirme kapalı. pretrained=False iken boş sözlük döner (final: bütün ağırlıklar checkpoint'ten). loaded: yüklenen
    (dosya, durum sözlüğü) kaydı (anahtar raporu için)."""
    import torch
    import torch.hub
    import torch.utils.model_zoo
    orig_load, orig_url = torch.load, torch.hub.load_state_dict_from_url

    def redirected(f, *a, **k):
        name = os.path.basename(str(f))
        if name not in PRETRAIN:
            raise RuntimeError(f'beklenmeyen torch.load yolu kurucuda: {f}')
        if not pretrained and name not in ALWAYS_LOAD:
            return {}
        path = PRETRAIN[name]
        if not os.path.exists(path):
            raise SystemExit(f'ön-eğitim dosyası yok: {path}\n'
                             + ('MiT-B4 ImageNet ağırlığı (SegFormer mit_b4.pth) indirilip bu yola konmalı.'
                                if name == 'mit_b4.pth' else ''))
        sd = orig_load(path, map_location='cpu', weights_only=True)
        if name == 'HarD-MSEG-best.pth':
            sd = sd['model']                                 # run_retrain.py best.pth biçimi
        loaded.append((path, sd))
        return sd

    def blocked(*a, **k):
        raise RuntimeError('internetten ağırlık indirme kapalı (yerel dosya kullanılmalı)')

    torch.load = redirected
    torch.hub.load_state_dict_from_url = torch.utils.model_zoo.load_url = blocked
    try:
        yield
    finally:
        torch.load = orig_load
        torch.hub.load_state_dict_from_url = torch.utils.model_zoo.load_url = orig_url


def _build_meganet(repo, loaded):
    """MEGANet-ResNet EGANetModel (resmi sınıf, değiştirilmeden). Kurucudaki models.resnet34(pretrained=True) →
    torchvision resnet34(weights=None) + aynı ImageNet dosyası yerelden strict yükleme (local_pretrain; internetten
    indirme yok). CUDA yoksa (yalnız CPU denemesi) resmi gauss_kernel(cuda=True) CPU'da tutulur; GPU'da resmi kod
    aynen çalışır. Döndürür: (model, omurga anahtar raporu | None)."""
    import torch
    import torchvision.models as tvm
    import lib.modules as M
    import lib.EGANet as E
    for mod in (M, E):
        assert os.path.abspath(mod.__file__).startswith(repo + os.sep), f'yanlış modül: {mod.__file__}'
    orig_gk = M.__dict__.setdefault('_orig_gauss_kernel', M.gauss_kernel)
    M.gauss_kernel = orig_gk if torch.cuda.is_available() else (lambda channels=3, cuda=True: orig_gk(channels, False))

    class _LocalResNet:
        @staticmethod
        def resnet34(pretrained=False, **kw):
            net = tvm.resnet34(weights=None, **kw)
            if pretrained:
                sd = torch.load('resnet34-b627a593.pth')        # local_pretrain → PRETRAIN yerel dosya
                if sd:
                    net.load_state_dict(sd, strict=True)
            return net

    E.models = _LocalResNet
    model = E.EGANetModel()
    if not loaded:
        return model, None
    path, sd = loaded[-1]
    pre = {'conv1.': 'net.encoder1_conv.', 'bn1.': 'net.encoder1_bn.', 'layer1.': 'net.encoder2.',
           'layer2.': 'net.encoder3.', 'layer3.': 'net.encoder4.', 'layer4.': 'net.encoder5.'}
    msd = model.state_dict()
    bkeys = [k for k in msd if k.startswith(tuple(pre.values()))]
    mapped = {}
    for k in sd:
        p = next((p for p in pre if k.startswith(p)), None)
        if p is not None:
            mapped[pre[p] + k[len(p):]] = k
    common = [k for k in mapped if k in msd]
    inv = {v: k for k, v in mapped.items()}
    with torch.no_grad():
        diff = max((float((msd[k].float() - sd[mapped[k]].float()).abs().max()) for k in common),
                   default=float('nan'))
    # eksik: yalnız BN num_batches_tracked olmalı (eski biçim dosya; torchvision de aynı dosyayı böyle yükler)
    # fazla: EGANet'te kullanılmayan anahtarlar (fc.*: sınıflandırıcı)
    return model, dict(file=path, ckpt_keys=len(sd), backbone_keys=len(bkeys), loaded=len(common),
                       missing=[k for k in bkeys if k not in mapped], unexpected=[k for k in sd if inv.get(k) not in msd],
                       max_abs_diff=diff)


def _build_baasnet(pretrained):
    """BoundaryAwareAttentionUNet (resmi sınıf). Kurucudaki resnet101(weights=IMAGENET1K_V2) → aynı dosya yerelden
    strict yükleme (torchvision kendi yolundan indirmesin)."""
    import torch
    import torchvision.models as tvm
    import model as B
    assert os.path.abspath(B.__file__).startswith(os.path.join(REPOS, 'BAASNet') + os.sep), B.__file__

    def local_resnet101(weights=None):
        net = tvm.resnet101(weights=None)
        if pretrained:
            net.load_state_dict(torch.load('resnet101-cd907fc2.pth'), strict=True)   # local_pretrain → yerel dosya
        return net

    B.resnet101 = local_resnet101
    return B.BoundaryAwareAttentionUNet(num_classes=1, pretrained=True)


def build(method, pretrained=True):
    """Resmi model sınıfı. Döndürür: (model, omurga anahtar raporu)."""
    repo = import_official(method)
    loaded, rep = [], None
    with local_pretrain(pretrained, loaded):
        if method == 'hsnet':
            from lib.pvt import HSNet
            model = HSNet()
        elif method == 'cafenet':
            from lib.pvt import CAFE
            model = CAFE()
        elif method.startswith('pranetv2'):
            import torch
            import lib.pranet as P
            import utils.dataloader  # noqa: F401  (içe aktarınca cudnn kapatıyor; aşağıda geri açılır)
            # Res2Net kurucusu pretrained=True iken torch.load + strict yükleme yapar; ön-eğitimsiz kurulumda
            # (bellek yoklaması / final) pretrained=False'a çevrilir. Asıl işlev saklanır: sonraki kurulumlar etkilenmesin.
            orig = P.__dict__.setdefault('_orig_res2net50', P.res2net50_v1b_26w_4s)
            P.res2net50_v1b_26w_4s = orig if pretrained else (lambda pretrained=False, **k: orig(pretrained=False, **k))
            nc = RECIPES[method]['num_class']
            model = P.PraNet_V2(num_class=nc) if method == 'pranetv2_res' else P.PVT_PraNet_V2(num_class=nc)
            torch.backends.cudnn.enabled = True      # TARİF SAPMASI (yalnız hız), RECIPES['sapma']
        elif method == 'meganet_r34':
            model, rep = _build_meganet(repo, loaded)
        elif method == 'fcbformer':
            from Models.models import FCBFormer
            model = FCBFormer()
        elif method == 'baasnet':
            model = _build_baasnet(pretrained)
        elif method == 'hardmseg':
            from lib.HarDMSEG import HarDMSEG
            model = HarDMSEG()
        elif method == 'futranshnet':
            from lib.TransFuse import TransFuse_S
            model = TransFuse_S(pretrained=pretrained)
        else:
            _ssformer_model_stubs()
            from models.build_model import build as ss_build
            model = ss_build(model_name='mit_PLD_b4', class_num=1)
    if loaded and rep is None and not hasattr(model, 'backbone'):        # resmi kurucular strict yükledi
        rep = dict(files=[p for p, _ in loaded])
    elif loaded and rep is None:
        path, sd = loaded[-1]
        bsd = model.backbone.state_dict()
        common = [k for k in sd if k in bsd]
        diff = max((float((bsd[k].float() - sd[k].float()).abs().max()) for k in common), default=float('nan'))
        rep = dict(file=path, ckpt_keys=len(sd), backbone_keys=len(bsd), loaded=len(common),
                   missing=[k for k in bsd if k not in sd], unexpected=[k for k in sd if k not in bsd],
                   max_abs_diff=diff)
    return model, rep


# ---- resmi Train.py / MyTrain_med.py / utils.py'den birebir (utils.py thop içe aktardığı için kopyalandı) -----------
def structure_loss_pvt(pred, mask):
    """HSNet/Train.py (Polyp-PVT ile aynı; reduce='none' eski API → skaler ortalama)."""
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


def structure_loss_cafe(pred, mask):
    """CAFE-Net/train.py (reduction='mean')."""
    import torch
    import torch.nn.functional as F
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def structure_loss_pranetv2(pred, pred_bg, mask_fg, mask_bg):
    """PraNet-V2/binary_seg/MyTrain_med.py."""
    import torch
    import torch.nn.functional as F
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask_fg, kernel_size=31, stride=1, padding=15) - mask_fg)
    wbce = F.binary_cross_entropy_with_logits(pred, mask_fg, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    wbce2 = F.binary_cross_entropy_with_logits(pred_bg, mask_bg, reduction='none')
    wbce2 = (weit * wbce2).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask_fg) * weit).sum(dim=(2, 3))
    union = ((pred + mask_fg) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou + 0.8 * wbce2).mean()


def structure_loss_fu(pred, mask):
    """Fu-TransHNet/train_isic.py."""
    import torch
    import torch.nn.functional as F
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def update_w(loss1, loss2, loss3, n):
    """Fu-TransHNet/train_isic.py."""
    e = math.e
    w1 = (e ** (-loss1 / n)) / ((e ** (-loss1 / n)) + (e ** (-loss2 / n)) + (e ** (-loss3 / n)))
    w2 = (e ** (-loss2 / n)) / ((e ** (-loss1 / n)) + (e ** (-loss2 / n)) + (e ** (-loss3 / n)))
    w3 = (e ** (-loss3 / n)) / ((e ** (-loss1 / n)) + (e ** (-loss2 / n)) + (e ** (-loss3 / n)))
    return w1, w2, w3


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
# ---------------------------------------------------------------------------------------------------------------


def method_loss(method, outs, gts):
    """Resmi eğitim betiğindeki kayıp (çıktı sırası resmi)."""
    import torch
    if method == 'hsnet':
        return sum(structure_loss_pvt(p, gts) for p in outs)                  # loss_P1 + … + loss_P4
    if method == 'cafenet':
        return sum(structure_loss_cafe(p, gts) for p in outs)                 # loss_P1 + … + loss_P5
    if method.startswith('pranetv2'):
        bg = 1 - gts
        fg2, fg3, fg4, fg5, bg2, bg3, bg4, bg5 = outs
        loss5 = structure_loss_pranetv2(fg2, bg2, gts, bg)
        loss4 = structure_loss_pranetv2(fg3, bg3, gts, bg)
        loss3 = structure_loss_pranetv2(fg4, bg4, gts, bg)
        loss2 = structure_loss_pranetv2(fg5, bg5, gts, bg)
        return loss2 + loss3 + loss4 + loss5
    if method == 'fcbformer':                                                # Dice_loss(output, target) + BCE_loss(sigmoid(output), target)
        from Metrics.losses import SoftDiceLoss
        return SoftDiceLoss()(outs, gts) + torch.nn.BCELoss()(torch.sigmoid(outs), gts)
    if method == 'baasnet':                                                  # CombinedLoss(dice 1, bce 1, kenar 0.5), resmi sınıf
        from loss import CombinedLoss
        return CombinedLoss(dice_weight=1.0, bce_weight=1.0, edge_weight=0.5)(outs, gts)
    if method == 'hardmseg':
        return structure_loss_pvt(outs, gts)
    if method == 'futranshnet':                                              # train_isic.py train(), n = 1
        n = 1
        loss4, loss3, loss2 = (structure_loss_fu(m, gts) for m in outs)      # lateral_map_4, _3, _2 = model(images)
        w1, w2, w3 = update_w(loss2, loss3, loss4, n)
        return w1 * loss2 + w2 * loss3 + w3 * loss4 + n * (w1 * math.log(w1, math.e) + w2 * math.log(w2, math.e)
                                                           + w3 * math.log(w3, math.e))
    if method == 'meganet_r34':                                              # criteria_loss(predicts, gts), resmi sınıf
        from utils.loss import DeepSupervisionLoss
        return DeepSupervisionLoss()(outs, gts)
    from utils.loss import dice_bce_loss                                     # SSFormer: resmi kayıp, değiştirilmeden
    return dice_bce_loss(torch.sigmoid(outs), gts)                            # out = nn.Sigmoid()(out); loss(out, label)


def set_lr(method, optimizer, epoch):
    """Epoch başında resmi LR kuralı (resmi döngüdeki yeriyle aynı: eğitimden önce)."""
    r = RECIPES[method]
    if 'plateau' in r:                                       # ReduceLROnPlateau, epoch sonunda doğrulamayla (train())
        return
    if method == 'meganet_r34':                              # LambdaLR(1 - (e / epoch) ** power), kapalı biçim
        for g in optimizer.param_groups:
            g['lr'] = r['lr'] * (1.0 - pow(epoch / r['poly_epochs'], r['power']))
        return
    if method == 'ssformer_l':
        lr = 0.0 if epoch == 0 else r['lr'] * r['lr_gamma'] ** ((epoch - 1) // r['lr_step'])
        for g in optimizer.param_groups:
            g['lr'] = lr
        return
    if method.startswith('pranetv2'):                       # adjust_lr(optimizer, 1e-4, epoch, 0.1, 50)
        for g in optimizer.param_groups:
            g['lr'] *= 0.1 ** (epoch // 50)
        return
    if epoch in r['lr_drop']:                                # HSNet adjust_lr(opt, 0.5) / CAFE adjust_lr_d(…, 0.5)
        for g in optimizer.param_groups:
            g['lr'] *= r['lr_drop'][epoch]


def make_optimizer(method, model):
    import torch
    r = RECIPES[method]
    if r['optimizer'] == 'SGD':                              # MEGANet train.py
        return torch.optim.SGD(model.parameters(), lr=r['lr'], momentum=r['momentum'], weight_decay=r['wd'])
    if r['optimizer'] == 'AdamW':
        return torch.optim.AdamW(model.parameters(), r['lr'], weight_decay=r['wd'])
    return torch.optim.Adam(model.parameters(), r['lr'], betas=r.get('betas', (0.9, 0.999)))


class DropName:
    """BAASNet veri kümesi (görüntü, maske, ad) döndürür; eğitim döngüsü (görüntü, maske) bekler."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i][:2]


def fu_npy(train_dir, out, size=352):
    """Fu-TransHNet/process_train.py process_and_save: 352×352 uint8 dizileri (görüntü RGB, maske gri, cv2 bilineer)."""
    import cv2
    pi, pm = os.path.join(out, 'data_train.npy'), os.path.join(out, 'mask_train.npy')
    if os.path.exists(pi) and os.path.exists(pm):
        return pi, pm
    names = sorted(os.listdir(os.path.join(train_dir, 'images')))
    imgs = np.zeros((len(names), size, size, 3), np.uint8)
    masks = np.zeros((len(names), size, size), np.uint8)
    for i, n in enumerate(names):
        img = cv2.imread(os.path.join(train_dir, 'images', n))
        imgs[i] = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size))
        masks[i] = cv2.resize(cv2.imread(os.path.join(train_dir, 'masks', n), 0), (size, size))
    for p, a in ((pi, imgs), (pm, masks)):
        with open(p + '.tmp', 'wb') as f:
            np.save(f, a)
        os.replace(p + '.tmp', p)
    return pi, pm


def make_loader(method, train_dir, batch, workers):
    """Resmi veri yükleyici (resmi get_loader / CustomDataSet / SegDataset / SkinDataset, değiştirilmeden)."""
    r = RECIPES[method]
    img, gt = os.path.join(train_dir, 'images') + '/', os.path.join(train_dir, 'masks') + '/'
    if method == 'fcbformer':                                # Data/dataloaders.py get_dataloaders (eğitim dönüşümleri)
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from Data.dataset import SegDataset
        names = sorted(os.listdir(img))
        tin = transforms.Compose([transforms.ToTensor(), transforms.Resize((352, 352), antialias=True),
                                  transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
                                  transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        tgt = transforms.Compose([transforms.ToTensor(), transforms.Resize((352, 352), antialias=False),
                                  transforms.Grayscale()])    # antialias=False: resmi torchvision 0.10 davranışı
        ds = SegDataset([img + n for n in names], [gt + n for n in names], tin, tgt, hflip=True, vflip=True, affine=True)
        return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers)
    if method == 'baasnet':                                  # train.py: Datasetloader_ + train_transform
        from torch.utils.data import DataLoader
        import albumentations as A
        from dataloader import Datasetloader_, train_transform
        assert isinstance(train_transform.transforms[0], A.Resize)
        tf = A.Compose([A.Resize(r['size'], r['size'])] + list(train_transform.transforms)[1:])   # yalnız ölçek değişti
        ds = DropName(Datasetloader_(root_dir=train_dir, transform=tf))
        return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers)
    if method == 'hardmseg':                                 # Train.py get_loader, augmentation=False varsayılan
        from utils.dataloader import get_loader
        return get_loader(img, gt, batchsize=batch, trainsize=r['size'], num_workers=workers, augmentation=r['aug'])
    if method == 'futranshnet':                              # train_isic.py get_loader → SkinDataset(.npy)
        from torch.utils.data import DataLoader
        from utils.dataloader import SkinDataset
        pi, pm = fu_npy(train_dir, os.path.dirname(train_dir))   # runs/<ad>/data/ (git dışı)
        return DataLoader(SkinDataset(pi, pm), batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True,
                          persistent_workers=workers > 0)    # .npy işçilere taşınır; her epoch yeniden açılmasın
    if method in ('hsnet', 'cafenet'):
        from utils.dataloader import get_loader
        return get_loader(img, gt, batchsize=batch, trainsize=r['size'], num_workers=workers, augmentation=r['aug'])
    if method.startswith('pranetv2'):
        from utils.dataloader import get_loader
        return get_loader(img, gt, batchsize=batch, trainsize=r['size'], num_workers=workers)
    if method == 'meganet_r34':                              # train.py: get_loader(..., batchsize, trainsize), aug varsayılan
        from utils.dataloader import get_loader
        return get_loader(img, gt, batchsize=batch, trainsize=r['size'], num_workers=workers)
    import yaml
    from torch.utils.data import DataLoader
    from utils.my_dataset import CustomDataSet
    with open(os.path.join(REPOS, 'ssformer', 'configs', 'ssformer-L.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    tl = cfg['Train_transform_list']
    allowed = {'resize', 'random_scale_crop', 'random_flip', 'random_rotate', 'random_image_enhance',
               'random_dilation_erosion', 'tonumpy', 'normalize', 'totensor'}
    assert set(tl) <= allowed, f'beklenmeyen dönüşüm adı (CustomDataSet eval eder): {set(tl) - allowed}'
    assert cfg['dataset']['batch_size'] == r['batch'] and cfg['model']['model_name'] == 'mit_PLD_b4'
    ds = CustomDataSet(img, gt, transform_list=tl)
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers)


def train_step(method, model, optimizer, images, gts, dev, accum):
    """Bir resmi eğitim adımı. accum > 1 yalnız bellek yetmezse: batch accum parçaya bölünür, kayıplar parça
    payıyla ağırlıklanıp toplanır (batch ortalamalı kayıplarda tam batch gradyanının aynısı; BN istatistikleri
    parça üzerinden). Döndürür: rate = 1 kaybı (resmi kayıt gibi)."""
    import torch
    import torch.nn.functional as F
    r = RECIPES[method]
    rec = None
    for rate in r['rates']:
        optimizer.zero_grad()
        images_, gts_ = images.to(dev), gts.to(dev).float()   # SkinDataset maskeleri float64
        if rate != 1:
            trainsize = int(round(r['size'] * rate / 32) * 32)
            images_ = F.interpolate(images_, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            gts_ = F.interpolate(gts_, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
        total, n = 0.0, images_.shape[0]
        for xi, yi in zip(images_.chunk(accum), gts_.chunk(accum)):
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled='amp' in r):
                loss = method_loss(method, model(xi), yi) * (xi.shape[0] / n)
            loss.backward()
            total += loss.item()
        if r['clip'] is not None:
            clip_gradient(optimizer, r['clip'])
        if 'clip_norm' in r:
            torch.nn.utils.clip_grad_norm_(model.parameters(), r['clip_norm'])
        optimizer.step()
        if rate == 1:
            rec = total
    return rec


def combine(method, out, size):
    """Resmi test betiğindeki çıktı → GT boyutunda olasılık (min-max öncesi)."""
    import torch.nn.functional as F
    if method == 'ssformer_l':                               # test.py: sigmoid → ölçekle
        return F.interpolate(out.sigmoid(), size=size, mode='bilinear', align_corners=False)
    if method == 'hsnet':
        out = out[0] + out[1] + out[2] + out[3]
    elif method in ('fcbformer', 'baasnet', 'hardmseg'):     # tek çıktı (BAASNet eval kipinde yalnız ana çıktı)
        pass
    elif method == 'futranshnet':                            # test_isic.py: res1, res2, res3 = model(image)
        out = 0.34230294885896795 * out[2] + 0.3268978957442428 * out[1] + 0.3307991553967893 * out[0]
    elif method == 'cafenet':
        out = out[0]
    elif method == 'meganet_r34':                            # test.py / eval_meganet.py: predicts[0] (out1)
        out = out[0]
    else:
        out = out[0] + out[1] + out[2] + out[3]
    return F.interpolate(out, size=size, mode='bilinear', align_corners=False).sigmoid()


def input_transform(method, size):
    """Resmi test ön işlemesi: PIL görüntü → tensör."""
    import torchvision.transforms as T
    if method == 'fcbformer':                                # Data/dataloaders.py transform_input4test
        return T.Compose([T.ToTensor(), T.Resize((352, 352), antialias=True),
                          T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if method == 'baasnet':                                  # dataloader.py val_transform: 0-255, normalizasyonsuz
        import albumentations as A
        from dataloader import val_transform
        assert isinstance(val_transform.transforms[0], A.Resize)
        tf = A.Compose([A.Resize(size, size)] + list(val_transform.transforms)[1:])                # yalnız ölçek değişti
        return lambda img: tf(image=np.array(img))['image'].float()
    if method == 'futranshnet':                              # process_test.py cv2.resize + test_dataset dönüşümü
        import cv2
        norm = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return lambda img: norm(cv2.resize(np.array(img), (size, size)))
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def predict(method, model, root, dev, size=352):
    """Resmi test ön işlemesi → {stem: (uint8 harita, gt bool)}; harita GT boyutunda, harita başına min-max
    (run_polyppvt.predict gibi)."""
    import torch
    from PIL import Image
    tf = input_transform(method, size)
    gdir, idir = os.path.join(root, 'masks'), os.path.join(root, 'images')
    gts = {os.path.splitext(f)[0]: f for f in os.listdir(gdir) if f.lower().endswith(IMG_EXT)}
    out = {}
    model.eval()
    with torch.no_grad():
        for f in sorted(os.listdir(idir)):
            stem = os.path.splitext(f)[0]
            if not f.lower().endswith(IMG_EXT) or stem not in gts:
                continue
            with open(os.path.join(idir, f), 'rb') as fh:
                img = Image.open(fh).convert('RGB')
            with open(os.path.join(gdir, gts[stem]), 'rb') as fh:
                gt = np.asarray(Image.open(fh).convert('L'), np.float32)
            gt /= (gt.max() + 1e-8)
            res = combine(method, model(tf(img).unsqueeze(0).to(dev)), gt.shape)
            res = res.float().cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            out[stem] = (np.round(res * 255).astype(np.uint8), gt > 0.5)
    assert len(out) == len(gts), f'{root}: {len(out)}/{len(gts)} harita'
    return out


def val_score(method, model, val_root, dev):
    return float(np.mean([pranet_dice(m, g) for m, g in predict(method, model, val_root, dev).values()]))


def probe_memory(method, dev):
    """Resmi batch en büyük ölçekte sığıyor mu (E1 kuralı: ayrılan bellek ≤ 14 GB)? Sığmazsa gradyan biriktirme
    (aynı etkin batch). Döndürür: (accum, tepe GB)."""
    import torch
    r = RECIPES[method]
    s = int(round(r['size'] * max(r['rates']) / 32) * 32)
    for accum in (a for a in (1, 2, 4) if a >= r.get('min_accum', 1)):
        model, _ = build(method, pretrained=False)
        model = model.to(dev).train()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled='amp' in r):
                out = model(torch.randn(r['batch'] // accum, 3, s, s, device=dev))
            outs = out if isinstance(out, (tuple, list)) else (out,)
            sum(o.float().mean() for o in outs).backward()
            peak = torch.cuda.max_memory_reserved() / 2 ** 30
        except torch.cuda.OutOfMemoryError:
            peak = float('inf')
        del model
        torch.cuda.empty_cache()
        if peak <= MEM_LIMIT_GB:
            return accum, peak
    raise SystemExit(f'{method}: batch {r["batch"]} 4 parçaya bölününce de {s} px sığmıyor')


def _rng_state():
    import torch
    st = np.random.get_state()
    s = dict(torch=torch.get_rng_state(), numpy=[st[0], st[1].tolist(), int(st[2]), int(st[3]), float(st[4])],
             python=random.getstate())
    if torch.cuda.is_available():
        s['cuda'] = torch.cuda.get_rng_state_all()
    return s


def _set_rng_state(s):
    import torch
    torch.set_rng_state(s['torch'])
    n = s['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    py = s['python']
    random.setstate((py[0], tuple(py[1]), py[2]))
    if 'cuda' in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s['cuda'])


def _save(obj, path):
    import torch
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)                          # yarım yazılmış checkpoint kalmasın


def train(method, opt):
    import torch
    r = RECIPES[method]
    run = opt.run_dir
    os.makedirs(run, exist_ok=True)
    dirs, split = subset_dirs(run, opt.subset)
    dev = torch.device(opt.device)
    last_p, best_p, rows_p = (os.path.join(run, n) for n in ('last.pth', 'best.pth', 'epochs.jsonl'))
    if os.path.exists(rows_p) and not os.path.exists(last_p):
        raise SystemExit(f'{rows_p} var ama last.pth yok: elle incelenmeli')
    resume = torch.load(last_p, map_location='cpu', weights_only=True) if os.path.exists(last_p) else None
    if resume is not None and resume['method'] != method:
        raise SystemExit(f'{last_p} başka yönteme ait: {resume["method"]}')

    if resume is not None:
        accum, peak = resume['accum'], float('nan')
    elif dev.type == 'cuda':
        accum, peak = probe_memory(method, dev)
    else:
        accum, peak = r.get('min_accum', 1), float('nan')
    seed = r['seed']
    torch.manual_seed(seed), np.random.seed(seed), random.seed(seed)
    if dev.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model, rep = build(method, pretrained=not opt.no_pretrain)
    model = model.to(dev)
    optimizer = make_optimizer(method, model)
    sched = None
    if 'plateau' in r:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', **r['plateau'])
    loader = make_loader(method, dirs['train'], r['batch'], r['workers'] if opt.workers is None else opt.workers)
    start, end = r['epochs']
    if opt.epochs:
        end = start + opt.epochs
    rows, best_epoch, best_val = [], None, -1.0
    if resume is None:
        sap = list(r['sapma']) + ([f'gradyan biriktirme ×{accum} (mikro batch {r["batch"] // accum}; etkin batch '
                                   f'{r["batch"]} aynı, BN istatistikleri mikro batch üzerinden)'] if accum > 1 else [])
        log(run, f'{method}: resmi kod {r["repo"]} ({r["src"]}); eğitim {len(split["train"])}, doğrulama '
                 f'{len(split["val"])}; {r["optimizer"]} lr {r["lr"]} wd {r["wd"]}, batch {r["batch"]}, '
                 f'ölçekler {r["rates"]}, kırpma {r["clip"]}, norm kırpma {r.get("clip_norm")}, betas {r.get("betas")}, '
                 f'plato {r.get("plateau")}, epoch {start}..{end - 1}, {len(loader)} adım/epoch, '
                 f'en büyük ölçekte ayrılan bellek {peak:.1f} GB; kayıp: {r["loss"]}')
        if method == 'hsnet':
            log(run, f'hsnet artırma bayrağı {r["aug"]!r} → dataloader {loader.dataset.augmentations!r} == "True"? '
                     f'{loader.dataset.augmentations == "True"} (resmi kodda da etkisiz)')
        if method == 'meganet_r34':
            log(run, f'meganet artırma: dataloader augmentations={loader.dataset.augmentations!r} (== True → '
                     f'RandomRotate(0–360, p=0.5) + V/H çevirme); SGD momentum {r["momentum"]}; poly '
                     f'lr = {r["lr"]}·(1 − e/{r["poly_epochs"]})^{r["power"]}'
                     + ('' if torch.cuda.is_available() else '; CUDA yok → gauss_kernel CPU (yalnız deneme)'))
        if rep is not None and 'files' in rep:
            log(run, f'ön-eğitim dosyaları (resmi kurucu yükledi): {rep["files"]}')
        elif rep is not None:
            log(run, f'omurga {rep["file"]}: yüklenen {rep["loaded"]}/{rep["backbone_keys"]} anahtar, eksik '
                     f'{len(rep["missing"])} {rep["missing"][:6]}, fazla {len(rep["unexpected"])} '
                     f'{rep["unexpected"][:6]}, en büyük fark {rep["max_abs_diff"]:.2e}')
        for s in sap:
            log(run, f'TARİF SAPMASI: {s}')
    else:
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        if sched is not None:
            sched.load_state_dict(resume['sched'])
        rows, best_epoch, best_val = resume['rows'], resume['best_epoch'], resume['best_val']
        with open(rows_p, 'w', encoding='utf-8') as f:       # günlüğü checkpoint'le eşitle (kesinti sonrası)
            f.writelines(json.dumps(x) + '\n' for x in rows)
        bck = torch.load(best_p, map_location='cpu', weights_only=True) if os.path.exists(best_p) else None
        if bck is None or bck['epoch'] != best_epoch:
            if best_epoch != resume['epoch']:
                raise SystemExit(f'best.pth (epoch {None if bck is None else bck["epoch"]}) last.pth kaydıyla '
                                 f'(en iyi {best_epoch}) uyuşmuyor')
            _save({'model': resume['model'], 'epoch': best_epoch, 'val': best_val}, best_p)
        start = resume['epoch'] + 1
        _set_rng_state(resume['rng'])
        log(run, f'DEVAM: epoch {resume["epoch"]} sonrasından (en iyi doğrulama epoch {best_epoch} {best_val:.4f})')
        del resume

    for epoch in range(start, end):
        set_lr(method, optimizer, epoch)
        model.train()
        t0, losses = time.time(), []
        for i, (images, gts) in enumerate(loader, start=1):
            losses.append(train_step(method, model, optimizer, images, gts, dev, accum))
            if opt.max_steps and i >= opt.max_steps:
                break
        t1 = time.time()
        v = val_score(method, model, dirs['val'], dev)
        lr_now = optimizer.param_groups[0]['lr']              # bu epoch'un LR'si (plato aşağıda düşürür)
        if sched is not None:
            sched.step(v)
        row = {'epoch': epoch, 'lr': lr_now, 'loss': float(np.mean(losses)), 'val': v,
               'train_min': (t1 - t0) / 60, 'val_min': (time.time() - t1) / 60}
        rows.append(row)
        improved = v > best_val
        if improved:
            best_epoch, best_val = epoch, v
        # sıra: last.pth → best.pth → epochs.jsonl (kesintide devam bunlardan tutarlı kurulur)
        _save({'method': method, 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'rows': rows, 'best_epoch': best_epoch, 'best_val': best_val, 'accum': accum, 'rng': _rng_state(),
               'sched': None if sched is None else sched.state_dict()}, last_p)
        if improved:
            _save({'model': model.state_dict(), 'epoch': epoch, 'val': v}, best_p)
        with open(rows_p, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        log(run, f'EPOCH {epoch:03d} lr {row["lr"]:.2e} loss {row["loss"]:.4f} val {v:.4f}'
                 f'{" *" if improved else ""} (en iyi {best_epoch} {best_val:.4f}) eğitim {row["train_min"]:.1f} dk '
                 f'doğrulama {row["val_min"]:.2f} dk')
        if opt.stop_after is not None and epoch >= opt.stop_after:
            log(run, f'DENEME: epoch {epoch} sonrası bilerek durduruldu (--stop-after)')
            return
    log(run, f'EĞİTİM BİTTİ: en iyi doğrulama epoch {best_epoch} ({best_val:.4f}), son epoch {rows[-1]["epoch"]}')


def final(method, opt):
    """best.pth (doğrulamayla seçilen) ve last.pth için 5 test setinde harita yaz (eval_external.py düzeni)."""
    import torch
    from PIL import Image
    run, dev = opt.run_dir, torch.device(opt.device)
    dirs, _ = subset_dirs(run, opt.subset)
    out_root = opt.maps_root or DL
    sets = SETS[:opt.sets] if opt.sets else SETS
    for tag, fname, sub in (('val', 'best.pth', f'maps_{method}_retrain'),
                            ('last', 'last.pth', f'maps_{method}_retrain_last')):
        ck = torch.load(os.path.join(run, fname), map_location='cpu', weights_only=True)
        model, _ = build(method, pretrained=False)
        model.load_state_dict(ck['model'])                    # strict: eksik/fazla anahtar hata
        model = model.to(dev)
        v = val_score(method, model, dirs['val'], dev)
        log(run, f'FINAL {tag}: epoch {ck["epoch"]}, doğrulama yeniden ölçüldü {v:.4f}')
        for ds in sets:
            d = os.path.join(out_root, sub, ds)
            os.makedirs(d, exist_ok=True)
            maps = predict(method, model, os.path.join(TEST, ds), dev)
            for stem, (m, _) in maps.items():
                Image.fromarray(m).save(os.path.join(d, stem + '.png'))
            log(run, f'FINAL {tag} {ds}: {len(maps)} harita → {d}')
        del model
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    log(run, f"eval_external.py SOURCES önerisi: '{method} (retrain)': ('maps_{method}_retrain', 'bizim protokol, "
             f"doğrulamayla seçilen epoch, run_retrain.py'), '{method} (retrain, son)': ('maps_{method}_retrain_last', …)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('method', choices=list(RECIPES))
    p.add_argument('stage', choices=['train', 'final'])
    p.add_argument('--run-dir', default='')
    p.add_argument('--device', default='cuda')
    # yalnız deneme (smoke) seçenekleri
    p.add_argument('--subset', type=int, default=0, help='deneme: ilk N eğitim/doğrulama görüntüsü')
    p.add_argument('--max-steps', type=int, default=0, help='deneme: epoch başına en çok adım')
    p.add_argument('--epochs', type=int, default=0, help='deneme: resmi başlangıçtan itibaren epoch sayısı')
    p.add_argument('--stop-after', type=int, default=None, help='deneme: bu epoch sonrası kesintiyi taklit et')
    p.add_argument('--workers', type=int, default=None, help='deneme: DataLoader işçi sayısı (resmi değer yerine)')
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--no-pretrain', action='store_true', help='deneme: omurga ön-eğitimi yüklenmez')
    p.add_argument('--maps-root', default='', help='deneme: harita kök klasörü (varsayılan downloads/)')
    p.add_argument('--sets', type=int, default=0, help='deneme: ilk N test seti')
    p.add_argument('--hardmseg-best', default='', help='deneme: Fu-TransHNet için HarD-MSEG-best.pth yerine bu dosya')
    opt = p.parse_args()
    if opt.hardmseg_best:
        PRETRAIN['HarD-MSEG-best.pth'] = opt.hardmseg_best
    opt.run_dir = os.path.abspath(opt.run_dir or os.path.join(ROOT, 'runs', f'retrain_{opt.method}'))
    if opt.threads:
        import torch
        torch.set_num_threads(opt.threads)
    {'train': train, 'final': final}[opt.stage](opt.method, opt)


if __name__ == '__main__':
    main()
