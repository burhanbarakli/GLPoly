"""Karşılaştırılan yöntemlerin hesaplama maliyeti: parametre, FLOP, FPS → revizyon/maliyet_rakipler.md.

Parametre/FLOP yalnızca CPU'da (CUDA_VISIBLE_DEVICES=-1, 2 iş parçacığı). Sayaç tools/cost.py ile birebir aynı: eval kipi,
torch.no_grad, FlopCounterMode(display=False), 352×352, batch 1, GFLOP = get_total_flops() / 1e9, GMAC = GFLOP / 2.
cost.py da FLOP'u CPU'da sayar; rakamlar revizyon/maliyet.md ile doğrudan karşılaştırılabilir.

FPS yalnızca --fps ile ölçülür, FLOP bittikten sonra ayrı bir GPU sürecinde (--fps-worker): CPU süreci CUDA'yı hiç görmez,
M2SNet/MEGANet'in resmi .cuda() çağrıları da GPU sürecinde değiştirilmeden çalışır. Ölçüm cost.py fps() işlevinin kendisi
(fp32, cudnn.benchmark, 50 ısınma + 200 ölçüm, synchronize). GPU boşken çalıştırılmalı; bunu betik denetlemez.

"test zamanı" satırları: GLPoly'de yalnızca MP1 için gereken grafik (to_test_time; lib/glpoly.py değişmez).

Modeller ön-eğitimsiz kurulur (FCBFormer/Fu-TransHNet omurgası ALWAYS_LOAD ile yüklenir; sayımı etkilemez). run_retrain.py'deki 9 yöntem onun build(pretrained=False) yolundan gelir (yerel
yönlendirme, mmcv/einops yer tutucuları, internet kapalı). Diğer depolar da aynı local_pretrain içinde kurulur; depolar
değiştirilmez, hazır betikleri çalıştırılmaz, yalnızca model sınıfları içe aktarılır, kurucudaki ön-eğitim çağrısı
çalışma anında pretrained=False'a çevrilir. Her yöntem kendi deposu yolda tek başınayken kurulur (lib/utils/model
paketleri depolar arasında çakışıyor). Kurulamayan yöntem 'ölçülemedi' ve nedeniyle yazılır, zorlanmaz.

FlopCounterMode'un toplamı değişmez; sınıfın _count_flops'u yalnızca kayıt için genişletilir (OpCensus): sayılmayan
işlemlerin çıktı eleman sayısı toplanır, böylece sayacın neyi kaçırdığı yöntem başına görülür.

    python -I -X utf8 glpoly/tools/cost_rivals.py           # -I: depoların paketleri glpoly'ninkilerle çakışmasın
    python -I -X utf8 glpoly/tools/cost_rivals.py --fps     # ek olarak GPU FPS (GPU boşken)
"""
import argparse
import collections
import contextlib
import gc
import importlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import types

sys.dont_write_bytecode = True                  # downloads/repos/ ve komşu depolara __pycache__ yazılmasın
CVD_ORIG = os.environ.get('CUDA_VISIBLE_DEVICES')
if '--fps-worker' not in sys.argv:              # torch'tan önce: GPU'yu yalnızca açıkça başlatılan FPS süreci görür
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

TOOLS = os.path.dirname(os.path.abspath(__file__))
GLPOLY = os.path.dirname(TOOLS)
ROOT = os.path.dirname(GLPOLY)
REPOS = os.path.join(ROOT, 'downloads', 'repos')
POLYP_PVT = os.path.join(os.path.dirname(ROOT), 'Polyp-PVT')
OUT = os.path.join(ROOT, 'revizyon', 'maliyet_rakipler.md')
SIZE = 352
GLPOLY_CFG = {'backbone': 'convnext_tiny', 'no_dec_norm': False}    # cost.py CONFIGS[0]
GLPOLY_REF = (40.63, 68.2)                                          # revizyon/maliyet.md
R2N_CFG = {'backbone': 'res2net50', 'no_dec_norm': False}           # cost.py CONFIGS[1]
BASE_CFG = {**GLPOLY_CFG, 'no_hfam': True, 'no_pam': True}          # cost.py CONFIGS[2]
TEST = ', test zamanı'                                              # test grafiği satırlarının ad eki
# tam grafik sağlaması: revizyon/maliyet.md ile birebir aynı olmalı (check_refs)
FULL_REFS = {'GLPoly (ConvNeXt-T)': GLPOLY_REF, 'GLPoly (Res2Net-50)': (37.33, 67.8),
             'Baseline (ConvNeXt-T, HFAM ×, PAM ×)': (37.20, 50.4)}

BASE_PATH = [p for p in sys.path if os.path.abspath(p or '.') not in (TOOLS, GLPOLY, ROOT)]
REPO_MODS = {'lib', 'utils', 'models', 'model', 'pretrain', 'res2net', 'CaraNet', 'mmcv', 'einops', 'Models'}

_spec = importlib.util.spec_from_file_location('run_retrain', os.path.join(TOOLS, 'run_retrain.py'))
rr = importlib.util.module_from_spec(_spec)
sys.modules['run_retrain'] = rr
_spec.loader.exec_module(rr)

# sayılmayan işlemler: ada göre gruplanır (ad sonundaki yerinde '_' atılır); kalanlar (reshape, cat...) maliyetsiz
OP_GROUPS = [
    ('norm', re.compile(r'norm')),
    ('interp', re.compile(r'upsample|interpolate|grid_sample')),
    ('pool', re.compile(r'pool')),
    ('aktivasyon', re.compile(r'^_?(relu|relu6|gelu|prelu|prelu_kernel|leaky_relu|elu|silu|mish|hardswish|hardsigmoid|'
                              r'hardtanh|sigmoid|tanh|softmax|log_softmax)$')),
    ('aritmetik', re.compile(r'^(add|sub|rsub|subtract|mul|multiply|div|divide|true_divide|abs|absolute|neg|pow|sqrt|'
                             r'exp|log|sum|mean|max|min|amax|amin|clamp|std|var)$')),
]
# doğrusal cebir / dikkat işlemi olup kayıtta bulunmayanlar: sonuca girseydi toplamı anlamlı biçimde değiştirirdi
SUSPECT = re.compile(r'attention|^dot$|^vdot$|^mv$|^ger$|^outer$|fft|mkldnn_linear|^bilinear$|^kron$')


class OpCensus(FlopCounterMode):
    """FlopCounterMode'un aynısı; toplam değişmez, ek olarak kayıtsız işlemlerin çıktı eleman sayısı tutulur."""

    def __init__(self):
        super().__init__(display=False)
        self.uncounted = collections.Counter()

    def _count_flops(self, func_packet, out, args, kwargs):
        if func_packet not in self.flop_registry:
            first = out[0] if isinstance(out, (tuple, list)) and len(out) else out
            self.uncounted[func_packet.__name__.rstrip('_')] += first.numel() if torch.is_tensor(first) else 0
        return super()._count_flops(func_packet, out, args, kwargs)


def self_check():
    """Sayaç gerçekten bu sürümde bekleneni sayıyor mu: kod çözücü conv'u, işlevsel conv, ConvNeXt'in 4B Linear'ı,
    PVT'nin 4B q @ k.transpose dikkati. Hepsi 2 FLOP/MAC olmalı (GMAC = GFLOP / 2 varsayımının dayanağı)."""
    def flops(fn, *a):
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            fn(*a)
        return fc.get_total_flops()

    assert flops(nn.Conv2d(3, 8, 3, padding=1), torch.randn(1, 3, 32, 32)) == 2 * 3 * 8 * 9 * 32 * 32
    assert flops(nn.Conv2d(8, 8, 3, padding=1, groups=8), torch.randn(1, 8, 32, 32)) == 2 * 8 * 1 * 9 * 32 * 32
    assert flops(lambda x: F.conv2d(x, torch.ones(4, 4, 3, 3), padding=1),
                 torch.randn(1, 4, 16, 16)) == 2 * 4 * 4 * 9 * 16 * 16
    assert flops(nn.Linear(10, 20), torch.randn(1, 7, 5, 10)) == 2 * 7 * 5 * 10 * 20
    assert flops(nn.Linear(10, 20), torch.randn(1, 7, 10)) == 2 * 7 * 10 * 20
    assert flops(torch.matmul, torch.randn(1, 8, 100, 64), torch.randn(1, 8, 64, 50)) == 2 * 8 * 100 * 64 * 50
    assert flops(torch.bmm, torch.randn(1, 100, 64), torch.randn(1, 64, 50)) == 2 * 100 * 64 * 50


def isolate(repo=None):
    """Önceki deponun paketlerini bellekten sil, yolda yalnızca bu depo kalsın (ad-alanı paketleri birleşmesin)."""
    for name in list(sys.modules):
        if name.split('.')[0] in REPO_MODS:
            del sys.modules[name]
    importlib.invalidate_caches()
    sys.path[:] = ([repo] if repo else []) + BASE_PATH


def check_origin(repo):
    """Yüklenen lib/utils/model modüllerinin hepsi bu depodan mı (yanlış depodan sayım olmasın)?"""
    for name, mod in list(sys.modules.items()):
        path = getattr(mod, '__file__', None)
        if name.split('.')[0] in REPO_MODS and path and not os.path.abspath(path).startswith(repo + os.sep):
            raise RuntimeError(f'yanlış modül: {name} → {path}')


def force_no_pretrain(mod, name):
    """Kurucudaki res2net...(pretrained=True) → pretrained=False (depo dosyası değişmez; çalışma anında)."""
    orig = getattr(mod, name)
    setattr(mod, name, lambda pretrained=False, **k: orig(pretrained=False, **k))


@contextlib.contextmanager
def cuda_noop():
    """M2SNet CNN1 kurucuda Parameter(...).cuda() çağırıyor; CUDA gizliyken hata verir. Gerçek .cuda() Parameter'dan düz
    tensör döndürür (modüle parametre olarak kaydolmaz), bu yüzden burada da detach: sayım GPU'daki kurulumla aynı olur.
    GPU (FPS) sürecinde resmi .cuda() olduğu gibi çalışır: düz tensör .to() ile taşınmayacağından zaten gereklidir."""
    if torch.cuda.is_available():
        yield
        return
    torch.Tensor.cuda = lambda self, *a, **k: self.detach()
    try:
        yield
    finally:
        del torch.Tensor.cuda


# ---- kurucular ----------------------------------------------------------------------------------------------------
class DroppedHead(nn.Module):
    """Testte gerekmeyen başlığın yerine: hesap yapmaz, parametresi yok, None döner. GLPoly.forward aynen çalışır."""

    def forward(self, x):
        return None


def to_test_time(model):
    """Yalnızca MP1 için gereken grafik. Sınır başlıklarının çıktısı forward'da başka yerde kullanılmaz, hepsi çıkar.
    HFAM açıkken ara maske başlıkları kalır (masks[-1] bir sonraki seviyenin HFAM girdisi); HFAM kapalıysa onlar da çıkar.
    forward kopyalanmadığı için grafik GLPoly ile uyumsuzlaşamaz; aynı ağırlıklarla MP1'in birebir aynı kaldığı doğrulanır."""
    model = model.eval()
    x = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad():
        full = model(x)[0][0]
    model.boundary_heads = nn.ModuleList(DroppedHead() for _ in model.boundary_heads)
    if not model.use_hfam:
        for i in range(1, len(model.mask_heads)):         # mask_heads[0] = MP1
            model.mask_heads[i] = DroppedHead()
    with torch.no_grad():
        masks, bounds = model(x)
    assert torch.equal(masks[0], full), 'test grafiği MP1 çıktısını değiştirdi'
    assert all(b is None for b in bounds) and all((m is None) != model.use_hfam for m in masks[1:])
    return model


def build_glpoly(cfg=GLPOLY_CFG, test_time=False):
    isolate(GLPOLY)
    from lib.glpoly import glpoly_from_args
    model = glpoly_from_args(cfg, pretrained=False)
    return to_test_time(model) if test_time else model, GLPOLY


def from_retrain(method):
    def build():
        isolate()
        model, _ = rr.build(method, pretrained=False)
        return model, sys.path[0]
    return build


def build_polyp_pvt():
    isolate(POLYP_PVT)
    with rr.local_pretrain(False, []):
        from lib.pvt import PolypPVT
        return PolypPVT(), POLYP_PVT


def build_caranet():
    repo = os.path.join(REPOS, 'CaraNet')
    isolate(repo)
    with rr.local_pretrain(False, []):
        import CaraNet as C
        force_no_pretrain(C, 'res2net101_v1b_26w_4s')
        return C.caranet(), repo


def build_cfanet():
    repo = os.path.join(REPOS, 'CFANet')
    isolate(repo)
    with rr.local_pretrain(False, []):
        from lib.model import CFANet
        return CFANet(), repo


def _msnet_family(cls):
    repo = os.path.join(REPOS, 'MSNet-M2SNet')
    isolate(repo)
    with rr.local_pretrain(False, []), cuda_noop():
        import model.miccai_msnet as N
        force_no_pretrain(N, 'res2net50_v1b_26w_4s')
        return getattr(N, cls)(), repo


def build_m2snet():
    model, repo = _msnet_family('M2SNet')
    for m in (model.conv_3, model.conv_5):              # gerçek .cuda() davranışı korunmuş olmalı
        assert 'weight' not in dict(m.named_parameters()) and not isinstance(m.weight, nn.Parameter)
    return model, repo


def build_pranet():
    repo = os.path.join(REPOS, 'PraNet')
    isolate(repo)
    with rr.local_pretrain(False, []):
        import lib.PraNet_Res2Net as P
        force_no_pretrain(P, 'res2net50_v1b_26w_4s')
        return P.PraNet(), repo


def build_sanet():
    repo = os.path.join(REPOS, 'SANet', 'src')
    isolate(repo)
    with rr.local_pretrain(False, []), contextlib.redirect_stdout(io.StringIO()):     # weight_init her modülü yazıyor
        from model import Model
        return Model(types.SimpleNamespace(snapshot=None)), repo


# (ad, kurucu, kurulum notu)
METHODS = [
    ('GLPoly (ConvNeXt-T)', build_glpoly, 'sağlama satırı (cost.py CONFIGS[0])'),
    ('GLPoly (ConvNeXt-T)' + TEST, lambda: build_glpoly(GLPOLY_CFG, True), 'sınır başlıkları yok'),
    ('GLPoly (Res2Net-50)', lambda: build_glpoly(R2N_CFG), 'sağlama satırı (cost.py CONFIGS[1])'),
    ('GLPoly (Res2Net-50)' + TEST, lambda: build_glpoly(R2N_CFG, True), 'sınır başlıkları yok'),
    ('Baseline (ConvNeXt-T, HFAM ×, PAM ×)', lambda: build_glpoly(BASE_CFG), 'sağlama satırı (cost.py CONFIGS[2])'),
    ('Baseline (ConvNeXt-T, HFAM ×, PAM ×)' + TEST, lambda: build_glpoly(BASE_CFG, True),
     'sınır ve MP2–MP5 başlıkları yok'),
    ('HSNet', from_retrain('hsnet'), 'run_retrain.build'),
    ('CAFE-Net', from_retrain('cafenet'), 'run_retrain.build'),
    ('PraNet-V2 (Res2Net-50)', from_retrain('pranetv2_res'), 'run_retrain.build'),
    ('PraNet-V2 (PVTv2-B2)', from_retrain('pranetv2_pvt'), 'run_retrain.build'),
    ('SSFormer-L', from_retrain('ssformer_l'), 'run_retrain.build; mmcv ConvModule/einops yer tutucu'),
    ('MEGANet-R34', from_retrain('meganet_r34'), 'run_retrain.build; gauss_kernel CPU (run_retrain yolu)'),
    ('FCBFormer', from_retrain('fcbformer'), 'run_retrain.build'),
    ('BAASNet', from_retrain('baasnet'), 'run_retrain.build; eğitildiği 352 çözünürlükte (resmi 512)'),
    ('Fu-TransHNet', from_retrain('futranshnet'), 'run_retrain.build; forward üç çıktıyı da hesaplıyor'),
    ('Polyp-PVT', build_polyp_pvt, '../Polyp-PVT lib/pvt.py'),
    ('CaraNet', build_caranet, 'res2net101 çağrısı pretrained=False'),
    ('CFA-Net', build_cfanet, 'CFANet(); kurucu ön-eğitim istemiyor'),
    ('M2SNet', build_m2snet, 'res2net50 pretrained=False; CNN1 .cuda() çalışma anında etkisiz'),
    ('MSNet', lambda: _msnet_family('MSNet'), 'res2net50 pretrained=False'),
    ('PraNet', build_pranet, 'res2net50 pretrained=False'),
    ('SANet', build_sanet, 'Model(snapshot=None); omurga torch.load → boş sözlük'),
]


def measure(model):
    """cost.py ile aynı sayım; ek olarak işlem dağılımı ve forward'da hiç çağrılmayan modüllerin parametreleri."""
    model = model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    called, hooks = set(), []
    for m in model.modules():
        hooks.append(m.register_forward_pre_hook(lambda mod, inp: called.add(id(mod))))
    x = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad(), OpCensus() as fc:
        model(x)
    for h in hooks:
        h.remove()
    gflops = fc.get_total_flops() / 1e9
    by_op = collections.Counter()
    for k, v in fc.get_flop_counts()['Global'].items():
        by_op['conv' if 'conv' in str(k) else 'mm'] += v / 1e9
    used = {id(p) for m in model.modules() if id(m) in called for p in m.parameters(recurse=False)}
    unused = collections.Counter()
    for name, p in model.named_parameters():
        if id(p) not in used:
            unused[name.rsplit('.', 1)[0]] += p.numel()
    groups = collections.Counter()
    other = collections.Counter()
    suspect = []
    for op, n in fc.uncounted.items():
        g = next((g for g, rx in OP_GROUPS if rx.search(op)), None)
        if SUSPECT.search(op):
            suspect.append(op)
        if g:
            groups[g] += n / 1e9
        else:
            other[op] += n / 1e9
    return dict(params=params, gflops=gflops, conv=by_op['conv'], mm=by_op['mm'], unused=sum(unused.values()) / 1e6,
                unused_top=unused.most_common(4), groups=groups, other=other, suspect=suspect,
                exact=(sum(p.numel() for p in model.parameters()), fc.get_total_flops()))


def check_refs(rows):
    """Tam grafik GLPoly satırları revizyon/maliyet.md ile aynı olmalı; test grafiği tam grafikten küçük olmalı."""
    by = {r['name']: r for r in rows}
    for name, (params, gflops) in FULL_REFS.items():
        full, test = by[name], by[name + TEST]
        assert 'params' in full and 'params' in test, f'{name}: ölçülemedi: {full.get("error")} {test.get("error")}'
        assert (round(full['params'], 2), round(full['gflops'], 1)) == (params, gflops), \
            f'{name}: {full["params"]:.4f} M, {full["gflops"]:.4f} GFLOP; beklenen {params} M, {gflops} GFLOP'
        assert test['params'] < full['params'] and test['gflops'] < full['gflops'], name


def fps_worker(path):
    """--fps-worker: CUDA görünür ayrı süreç. Sayım cost.py fps() ile yapılır; sonuç JSON olarak üst sürece döner."""
    spec = importlib.util.spec_from_file_location('glpoly_cost', os.path.join(TOOLS, 'cost.py'))
    cost = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cost)                       # lib.glpoly'yi de yükler; isolate() her kurulumda temizler
    assert torch.cuda.is_available(), 'FPS için CUDA gerekli'
    torch.manual_seed(0)
    dev = torch.device('cuda')
    xg = torch.randn(1, 3, SIZE, SIZE).to(dev)
    res = dict(device=torch.cuda.get_device_name(0), methods={})
    for name, builder, _ in METHODS:
        model = None
        try:
            model, repo = builder()
            check_origin(repo)
            model = model.eval().to(dev)
            torch.backends.cudnn.enabled = True         # PraNet-V2 utils.dataloader içe aktarınca kapatıyor
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True       # cost.py main() ile aynı
            res['methods'][name] = dict(fps=cost.fps(model, xg, None))
            print(f'{name}: fp32 {res["methods"][name]["fps"]:.1f} FPS', flush=True)
        except Exception as e:                          # ölçülemeyen yöntem zorlanmaz: neden tabloya yazılır
            msg = f'{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ""}'
            print(f'{name}: FPS ölçülemedi: {msg}', flush=True)
            res['methods'][name] = dict(error=msg.replace('|', '/'))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False)


def run_fps_worker():
    """FPS ayrı süreçte: üst süreç CUDA'yı gizli tutar (yukarıdaki FLOP'lar GPU kurulumundan etkilenmez)."""
    env = {k: v for k, v in os.environ.items() if k != 'CUDA_VISIBLE_DEVICES'}
    if CVD_ORIG is not None:
        env['CUDA_VISIBLE_DEVICES'] = CVD_ORIG
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'fps.json')
        subprocess.run([sys.executable, '-I', '-X', 'utf8', os.path.abspath(__file__), '--fps-worker', path],
                       env=env, check=True)
        with open(path, encoding='utf-8') as f:
            return json.load(f)


def fps_cell(fps, name):
    if fps is None:
        return 'ölçülmedi'
    r = fps['methods'].get(name, dict(error='sonuç yok'))
    return f'{r["fps"]:.1f}' if 'fps' in r else 'ölçülemedi'


def write_md(rows, fps=None, out=OUT):
    gl = next(r for r in rows if r['name'].startswith('GLPoly'))
    ok = [r for r in rows if 'params' in r]
    bad = [r for r in rows if 'params' not in r]
    fps_txt = ('FPS ölçülmedi.' if fps is None else
               f'FPS: {fps["device"]}, fp32, `cudnn.benchmark`, 50 ısınma + 200 ölçüm, `cuda.synchronize`, rastgele giriş '
               '(`cost.py fps()`); modeller ön-eğitimsiz kuruldu (FCBFormer ve Fu-TransHNet omurga ağırlığını kurucuda yükler).')
    head = ['# Karşılaştırılan yöntemlerin hesaplama maliyeti', '',
            f'`glpoly/tools/cost_rivals.py`; parametre ve FLOP yalnızca CPU\'da, PyTorch {torch.__version__}, giriş '
            f'{SIZE}×{SIZE}, batch 1, eval kipi. Sayaç `glpoly/tools/cost.py` ile aynı (`FlopCounterMode`, CPU\'da), '
            'GMAC = GFLOP / 2 (sayaç 1 MAC = 2 FLOP alır). Bütün başlıklar sayılır (yöntemin forward\'ı ne hesaplıyorsa); '
            f'istisna `test zamanı` satırları (bkz. Notlar). {fps_txt}',
            '',
            f'Sağlama: GLPoly (ConvNeXt-T) burada {gl.get("params", float("nan")):.2f} M, '
            f'{gl.get("gflops", float("nan")):.1f} GFLOP; `revizyon/maliyet.md`: {GLPOLY_REF[0]} M, {GLPOLY_REF[1]} GFLOP.',
            '', '## Tablo', '',
            '| Yöntem | Parametre (M) | GFLOP | GMAC | FPS fp32 | conv GFLOP | mm/bmm GFLOP | Çağrılmayan param. (M) | '
            'Kurulum |',
            '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        if 'params' in r:
            head.append(f'| {r["name"]} | {r["params"]:.2f} | {r["gflops"]:.1f} | {r["gflops"] / 2:.1f} | '
                        f'{fps_cell(fps, r["name"])} | {r["conv"]:.1f} | {r["mm"]:.1f} | {r["unused"]:.2f} | {r["note"]} |')
        else:
            head.append(f'| {r["name"]} | ölçülemedi | — | — | {fps_cell(fps, r["name"])} | — | — | — | {r["error"]} |')
    no_fps = [] if fps is None else [(k, v['error']) for k, v in fps['methods'].items() if 'error' in v]
    if bad:
        head += ['', '## Ölçülemeyenler', ''] + [f'- **{r["name"]}**: {r["error"]}' for r in bad]
    else:
        head += ['', 'İstenen yöntemlerin hepsi kuruldu ve ölçüldü; `ölçülemedi` satırı yok.']
    if no_fps:
        head += ['', '## FPS ölçülemeyenler', ''] + [f'- **{k}**: {v}' for k, v in no_fps]
    head += ['', '## Sayacın dışında kalanlar (yöntem başına)', '',
             'Sayılmayan işlemlerin çıktı eleman sayısı, milyar eleman (G); FLOP değildir, büyüklük fikri verir '
             '(eleman başına kabaca 1–2 işlem, bilineer ölçekleme için daha fazla). Oran sütunu: toplam eleman / GFLOP.', '',
             '| Yöntem | norm | interp | pool | aktivasyon | aritmetik | Oran (%) | Kayıtsız doğrusal cebir/dikkat |',
             '|---|---|---|---|---|---|---|---|']
    for r in ok:
        g = r['groups']
        head.append(f'| {r["name"]} | {g["norm"]:.3f} | {g["interp"]:.3f} | {g["pool"]:.3f} | {g["aktivasyon"]:.3f} | '
                    f'{g["aritmetik"]:.3f} | {100 * sum(g.values()) / r["gflops"]:.2f} | '
                    f'{", ".join(r["suspect"]) or "yok"} |')
    unused_lines = [f'- {r["name"]}: ' + ', '.join(f'`{k}` {v / 1e6:.2f} M' for k, v in r['unused_top'])
                    for r in ok if r['unused_top']]
    head += ['', '## Çağrılmayan parametreler', '',
             'Parametre sütunu istenen ölçüdür: `sum(p.numel())`, modelde kayıtlı her şey. "Çağrılmayan" sütunu bunun '
             '352×352 RGB forward\'ında hiç çağrılmayan modüllerdeki kısmıdır (en büyükleri). 2.05 M, Res2Net\'in '
             'kullanılmayan ImageNet sınıflandırıcısıdır (`fc`); bazı makaleler bunu saymaz, bu yüzden yayımlanmış '
             'parametre sayılarıyla fark çıkabilir. PraNet-V2\'deki `conv` yalnızca tek kanallı girişte çalışır. '
             'Fu-TransHNet\'te çağrılmayanlar HarDNet-MSEG içindeki `net.conv5`, `net.ra*` ve benzeri dallardır.', '']
    head += unused_lines
    head += ['', '## Notlar', '', *NOTES]
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(head) + '\n')
    print(f'yazıldı: {out}')


NOTES = [
    '- `test zamanı` satırları: yalnızca son maske MP1 için gereken GLPoly grafiği (`to_test_time`). Sınır başlıkları çıkar. '
    'HFAM açıkken ara maske başlıkları kalmak zorundadır (`masks[-1]` bir sonraki seviyenin HFAM girdisidir); HFAM kapalı '
    'baseline\'da MP2–MP5 başlıkları da çıkar. Aynı ağırlıklarla MP1 çıktısının birebir aynı kaldığı her kurulumda '
    'doğrulanır. Rakip yöntemlerde böyle bir kırpma yapılmadı: satırları forward\'larının hesapladığı her şeyi sayar, '
    'bu yüzden `test zamanı` satırları rakiplerle değil aynı modelin tam grafiğiyle karşılaştırılmalıdır.',
    '- FPS: rastgele girişle yalnızca model çağrısı (veri yükleme, ölçekleme, eşikleme yok); tek ölçüm. MEGANet resmi '
    '`gauss_kernel` ile ölçüldü: bu işlev her çağrıda çekirdeği CPU\'da kurup `.cuda()` ile taşır, bu yük FPS\'e dahildir.',
    '- Sayaç (bu sürümde kaynaktan okundu): `mm`, `addmm`, `bmm`, `baddbmm`, `_scaled_mm`, konvolüsyon ailesi ve dikkat '
    '(SDPA) işlemlerinin kayıtlı biçimleri. `matmul`/`linear` bunlara ayrıştırılarak sayılır; `self_check()` 4B Linear, '
    '4B matmul, gruplu conv ve işlevsel `F.conv2d` için beklenen değeri doğrular. Sayım dispatch düzeyinde olduğundan '
    'modül olmayan işlevsel conv\'lar da girer (MEGANet Gauss/Laplace süzgeçleri, M2SNet CNN1, GLPoly sabit süzgeçleri).',
    '- Sayılmayanlar: BN/LN/GN, interpolasyon, havuzlama, aktivasyon, eleman bazlı `+ * abs` (ör. SANet `out3*out4*out5`, '
    'ters dikkat dalları) ve bias. Yukarıdaki oran sütunu bunların hacminin conv/matmul yanında küçük kaldığını, yöntemler '
    'arası sıralamayı değiştirmediğini gösterir. Bu yöntemlerde kayıtsız doğrusal cebir ya da dikkat işlemi yok; özellikle '
    'CPU\'da kayıtsız kalan SDPA biçimi hiçbir satırda çalışmıyor (GLPoly `GlobalContext` bu yapılandırmada kapalı).',
    '- M2SNet\'in MSNet\'e göre fazlası (yaklaşık 32.6 GFLOP) tamamı birlerden oluşan, öğrenilmeyen `CNN1` süzgeçlerinden '
    'gelir: `F.conv2d` ile 64×64 kanal, gruplama yok, 3×3 ve 5×5. Bu sayaç hepsini sayar; modül kancalı sayaçlar saymaz. '
    'Depodaki `train.py` `MSNet` sınıfını eğitir; `M2SNet` aynı dosyada ayrı sınıftır.',
    '- Sayaçlar özel işlemlerde farklıdır: thop/ptflops/fvcore modül kancasıyla ya da başka kurallarla sayar; işlevsel '
    'conv, `bmm` ve dikkat matmul\'ünü kaçırabilir, 1 MAC = 1 FLOP alabilir. Makalelerin yayımladığı değerler bu tabloyla '
    'birebir karşılaştırılmamalı; tablodaki tüm satırlar aynı sayaçla ölçüldü.',
    '- Sabit süzgeçler parametreye girmez: M2SNet CNN1 ağırlıkları `Parameter(...).cuda()` sonucu düz tensör olduğundan '
    'resmi kodda da parametre değildir (FLOP\'a girer); GLPoly\'de Laplace/Gabor süzgeçleri buffer\'dır.',
    '- BAASNet eval kipinde yalnız ana çıktıyı döndürür, ama forward kenar dalını yine hesaplar (yaklaşık 4.6 GFLOP); '
    'diğer yöntemler gibi sayıma ve FPS\'e dahildir.',
    '- Depolar değiştirilmedi. Çalışma anı müdahaleleri yalnızca kurucuda: ön-eğitim kapatma, SSFormer `mmcv.cnn.ConvModule`'
    ' yer tutucusu (conv+BN+ReLU; resmi yapıyla aynı katmanlar), MEGANet `gauss_kernel` CPU, M2SNet `.cuda()` etkisizleştirme '
    '(bu son ikisi yalnızca CPU sayımında; FPS sürecinde resmi `.cuda()` yolu çalışır). Hiçbiri hesap grafiğini değiştirmez.',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fps', action='store_true', help='FLOP sayımından sonra ayrı süreçte GPU FPS de ölç')
    ap.add_argument('--out', default=OUT, help='çıktı md (deneme için; varsayılan revizyon/maliyet_rakipler.md)')
    ap.add_argument('--fps-worker', metavar='JSON', help='iç kullanım: GPU süreci, sonucu bu JSON dosyasına yazar')
    opt = ap.parse_args()
    if opt.fps_worker:
        fps_worker(opt.fps_worker)
        return
    torch.set_num_threads(2)                            # makine GPU eğitimi ve Drive kopyasıyla paylaşılıyor
    torch.manual_seed(0)
    assert not torch.cuda.is_available(), 'CUDA görünür: bu betik yalnızca CPU içindir'
    self_check()
    rows = []
    for name, builder, note in METHODS:
        t = time.time()
        try:
            model, repo = builder()
            check_origin(repo)
            r = measure(model)
            del model
            print(f'{name}: {r["exact"]} {r["params"]:.2f}M, {r["gflops"]:.1f} GFLOP (conv {r["conv"]:.1f}, mm '
                  f'{r["mm"]:.1f}), çağrılmayan {r["unused"]:.2f}M, kuşkulu {r["suspect"]}, '
                  f'diğer {[(k, round(v, 3)) for k, v in r["other"].most_common()]} ({time.time() - t:.0f} sn)', flush=True)
            rows.append(dict(name=name, note=note, **r))
        except Exception as e:                          # kurulamayan yöntem zorlanmaz: neden tabloya yazılır
            msg = f'{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ""}'
            print(f'{name}: ölçülemedi: {msg}', flush=True)
            rows.append(dict(name=name, note=note, error=msg.replace('|', '/')))
    isolate()
    check_refs(rows)
    write_md(rows, run_fps_worker() if opt.fps else None, opt.out)


if __name__ == '__main__':
    main()
