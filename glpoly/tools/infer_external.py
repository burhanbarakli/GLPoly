"""Yayımlanmış ağırlıklarla çıkarım (haritası yayımlanmamış yöntemler): PraNet-V2 (Res2Net-50 / PVTv2-B2), SSFormer-L.

Depoların hazır betikleri çalıştırılmaz; yalnızca model sınıfları içe aktarılır (downloads/repos/). Kurucuların gömülü
ön-eğitim yüklemeleri (sabit yol ya da internet) çıkarımda gereksizdir (bütün ağırlıklar checkpoint'ten gelir) ve devre
dışı bırakılır. Checkpoint weights_only=True ile yüklenir; anahtarlar birebir eşleşmek zorundadır (depoların kendi
test betikleri strict=False kullanıyor; burada eksik/fazla anahtar hata). Ön işleme ve çıktı, depoların test
betikleriyle aynı: PIL bilineer 352×352, /255, ImageNet normalizasyonu; PraNet-V2 için p2+p3+p4+p5 GT boyutuna
bilineer ölçeklenip sigmoid, SSFormer için sigmoid alınıp GT boyutuna bilineer ölçeklenir. Harita 0–255 gri olarak downloads/maps_<yöntem>/<set>/ altına
yazılır (float min-max, PraNet-V2 resmi testi gibi); değerlendirme eval_external.py ile (PraNet protokolü, 256 eşik).
Not: SSFormer'ın kendi test.py'si eşik taramasından önce 0.5'te ikilileştiriyor (fiilen tek eşik); bizim sayılarımız
sürekli haritalarla 256 eşik ortalaması olduğu için SSFormer makalesindeki tabloyla birebir aynı olmaz.
Mutlaka `python -I` ile çalıştırılmalı (depoların `lib`/`models` paketleri glpoly'ninkilerle çakışmasın).

    python -I tools/infer_external.py --method pranetv2_res|pranetv2_pvt|ssformer_l [--device cuda]
"""
import argparse
import contextlib
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DL = os.path.join(ROOT, 'downloads')
TEST = os.path.join(ROOT, 'dataset', 'TestDataset')
DATASETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
IMG_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
METHODS = {  # ad → (depo yolu, ağırlık)
    'pranetv2_res': ('repos/PraNet-V2/binary_seg', 'w_pranetv2_res/RES-V2.pth'),
    'pranetv2_pvt': ('repos/PraNet-V2/binary_seg', 'w_pranetv2_pvt/PVT-V2.pth'),
    'ssformer_l': ('repos/ssformer', 'w_ssformer_l/ssformer_L.pth'),
}


def stub_ssformer_deps():
    """SSFormer-L'nin kullandığı mmcv.cnn.ConvModule (conv → BN → ReLU; mmcv modül adları conv/bn/activate, norm
    varken conv'da bias yok) ve içe aktarılıp kullanılmayan einops için yer tutucular (mmcv/einops kurulu değil)."""
    class ConvModule(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_cfg=None,
                     act_cfg=dict(type='ReLU')):
            super().__init__()
            assert norm_cfg is None or norm_cfg.get('type') == 'BN', norm_cfg
            assert act_cfg is None or act_cfg.get('type') == 'ReLU', act_cfg
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=norm_cfg is None)
            self.bn = nn.BatchNorm2d(out_channels) if norm_cfg is not None else None
            self.activate = nn.ReLU(inplace=True) if act_cfg is not None else None

        def forward(self, x):
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            return self.activate(x) if self.activate is not None else x

    def unavailable(*a, **k):
        raise NotImplementedError('mmcv yer tutucusu: bu işlev kullanılmamalı')

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


@contextlib.contextmanager
def no_backbone_pretrain():
    """Kurucu içindeki torch.load (sabit yollu pvt_v2_b2.pth / mit_b4.pth) boş sözlük döndürür; checkpoint sonra yüklenir."""
    orig = torch.load
    torch.load = lambda *a, **k: {}
    try:
        yield
    finally:
        torch.load = orig


def build(method):
    repo, _ = METHODS[method]
    sys.path.insert(0, os.path.join(DL, repo))
    with no_backbone_pretrain():
        if method.startswith('pranetv2'):
            import lib.pranet as P
            orig = P.res2net50_v1b_26w_4s
            P.res2net50_v1b_26w_4s = lambda pretrained=False, **k: orig(pretrained=False, **k)  # internetten indirmesin
            return P.PraNet_V2(num_class=1) if method == 'pranetv2_res' else P.PVT_PraNet_V2(num_class=1)
        stub_ssformer_deps()
        from models.build_model import build as ss_build
        return ss_build(model_name='mit_PLD_b4', class_num=1)


def predict(method, model, x):
    if method.startswith('pranetv2'):
        p2, p3, p4, p5 = model(x)[:4]
        return p2 + p3 + p4 + p5
    return model(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=list(METHODS))
    p.add_argument('--device', default='cuda')
    opt = p.parse_args()
    torch.set_num_threads(2)                            # makine GPU eğitimi ve CPU testleriyle paylaşılıyor
    model = build(opt.method)
    sd = torch.load(os.path.join(DL, METHODS[opt.method][1]), map_location='cpu', weights_only=True)
    res = model.load_state_dict(sd, strict=False)
    # PraNet-V2 checkpoint'lerinde modelde karşılığı olmayan artık conv_last katmanı var (depoda hiç kullanılmıyor;
    # resmi test strict=False ile atlıyor); yalnızca bu ikisine izin verilir
    allowed = {'conv_last.weight', 'conv_last.bias'} if opt.method.startswith('pranetv2') else set()
    if res.missing_keys or set(res.unexpected_keys) - allowed:
        raise SystemExit(f'anahtar uyuşmazlığı: eksik {res.missing_keys[:5]} ({len(res.missing_keys)}), '
                         f'fazla {res.unexpected_keys[:5]} ({len(res.unexpected_keys)})')
    model = model.to(opt.device).eval()
    out_root = os.path.join(DL, f'maps_{opt.method}')
    with torch.no_grad():
        for ds in DATASETS:
            idir, gdir = os.path.join(TEST, ds, 'images'), os.path.join(TEST, ds, 'masks')
            gts = {os.path.splitext(f)[0]: f for f in os.listdir(gdir) if f.lower().endswith(IMG_EXT)}
            os.makedirs(os.path.join(out_root, ds), exist_ok=True)
            n = 0
            for f in sorted(os.listdir(idir)):
                stem = os.path.splitext(f)[0]
                if not f.lower().endswith(IMG_EXT) or stem not in gts:
                    continue
                img = Image.open(os.path.join(idir, f)).convert('RGB')
                with Image.open(os.path.join(gdir, gts[stem])) as g:
                    w, h = g.size
                a = (np.asarray(img.resize((352, 352), Image.BILINEAR), np.float32) / 255 - MEAN) / STD
                x = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).to(opt.device)
                y = predict(opt.method, model, x)
                if opt.method.startswith('pranetv2'):   # depoların test sırası: PraNet-V2 ölçekle → sigmoid,
                    m = torch.sigmoid(F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False))
                else:                                   # SSFormer sigmoid → ölçekle
                    m = F.interpolate(torch.sigmoid(y), size=(h, w), mode='bilinear', align_corners=False)
                m = m[0, 0].cpu().numpy()
                m = (m - m.min()) / (m.max() - m.min() + 1e-8)   # PraNet-V2 resmi testi gibi: float min-max, sonra uint8
                Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(out_root, ds, stem + '.png'))
                n += 1
            print(f'{opt.method} {ds}: {n}/{len(gts)} harita', flush=True)
            assert n == len(gts), f'{ds}: {n} harita, {len(gts)} GT'


if __name__ == '__main__':
    main()
