"""GLPoly: HFAM + PAM ile çift dallı tamamlayıcı dikkat, sınır farkındalıklı çok görevli decoder."""
import torch
import torch.nn as nn
from torchvision.transforms.functional import rgb_to_grayscale

from .backbones import build_backbone
from .context import GlobalContext
from .cues import EdgeCue, GaborConv, LearnableTexture
from .hfam import HFAM, PARTS
from .pam import PAM

# Seviye 1..5 için sabit kanal sayıları C_i
DEFAULT_CHANNELS = [64, 64, 128, 256, 512]


def conv_bn_relu(cin, cout, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResidualConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1))
        self.shortcut = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


def up_block(ch, dec_norm):
    """2× yukarı örnekleme; dec_norm ile ardından BN + ReLU (normalize kod çözücü çıktısı)."""
    up = nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)
    return nn.Sequential(up, nn.BatchNorm2d(ch), nn.ReLU(inplace=True)) if dec_norm else up


class DecoderStage(nn.Module):
    """Denklem (11)–(12) ve Şekil (dec.png): z̃_i = ResConv([s_i ‖ d_{i+1}]),  d_i = ConvTranspose2d(z̃_i).

    Şekildeki tek artık bağlantı birleştirilmiş girdiden gelir (ResidualConv'un 1×1 kısayolu, orj `Up`).
    Denklem (11)'deki ayrıca "+ s_i" ifadesi şekille çelişiyor; makalede düzeltilecek (PLAN.md §8).

    dec_norm=True (26 Ekim kararı): d_i = ReLU(BN(ConvTranspose(z̃_i))). Normalizasyonsuz kod çözücü yolunda
    ağırlıklar ve aktivasyonlar eğitim boyunca sınırsız büyüyor (başlıklardaki BN ölçeği sıfırladığı için kayıp
    bu büyüklüğü görmüyor); E3 fp16 koşusu bu yüzden epoch 46'da NaN'a gitti. False: makaledeki/orj yapısı.
    """

    def __init__(self, skip_ch, deep_ch, dec_norm=True):
        super().__init__()
        self.res = ResidualConv(skip_ch + deep_ch, skip_ch)
        self.up = up_block(skip_ch, dec_norm)

    def forward(self, s, d_deep):
        z = self.res(torch.cat([s, d_deep], dim=1))
        return self.up(z)


class Head(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.body = nn.Sequential(conv_bn_relu(cin, cin // 4), nn.Conv2d(cin // 4, 1, 1))

    def forward(self, x):
        return self.body(x)


class GLPoly(nn.Module):
    def __init__(self, backbone='res2net50', pretrained=True, use_hfam=True, use_pam=True,
                 n_gabor=4, laplace_level=1, hfam_residual=False, dec_norm=True, channels=DEFAULT_CHANNELS,
                 edge_op='laplace', texture_op='gabor', hfam_parts=PARTS, attn_refine=True, gc_block='none'):
        """edge_op / texture_op / hfam_parts / attn_refine: ablation'lar ve kontroller (varsayılan = makale).
        gc_block='attn': /32'de genel bağlam bloğu (darboğaz deneyi, makalede yok)."""
        super().__init__()
        self.use_hfam, self.use_pam = use_hfam, use_pam
        self.need_edge = use_hfam and 'edge' in hfam_parts
        C = channels

        self.encoder = build_backbone(backbone, pretrained)
        # Adjust Encoder (Denklem 1): 1×1 conv ile E_fi → E_i
        self.adjust = nn.ModuleList(conv_bn_relu(cin, c, k=1)
                                    for cin, c in zip(self.encoder.channels, C))
        self.gc = GlobalContext(C[4]) if gc_block == 'attn' else nn.Identity()

        self.edge_cue = EdgeCue(edge_op, laplace_level)
        self.gabor = GaborConv(n_filters=n_gabor) if texture_op == 'gabor' else LearnableTexture(n_filters=n_gabor)
        # Dikkat modülleri ve füzyon seviye 1..4'te (Şekil: Fuse1-4)
        self.hfam = nn.ModuleList(HFAM(c, hfam_residual, hfam_parts, attn_refine) for c in C[:4]) if use_hfam else None
        self.pam = nn.ModuleList(PAM(c, n_gabor, attn_refine) for c in C[:4]) if use_pam else None
        self.fuse = nn.ModuleList(conv_bn_relu(2 * c, c, k=1) for c in C[:4]) \
            if use_hfam and use_pam else None

        self.dec5 = nn.Sequential(conv_bn_relu(C[4], C[4]), up_block(C[4], dec_norm))
        # dec[i] seviye i+1: skip C[i], derin girdi C[i+1]
        self.dec = nn.ModuleList(DecoderStage(C[i], C[i + 1], dec_norm) for i in range(4))

        self.mask_heads = nn.ModuleList(Head(c) for c in C)
        self.boundary_heads = nn.ModuleList(Head(c) for c in C)

    def skip(self, i, e, prev_mask, edge, gabor):
        if not (self.use_hfam or self.use_pam):
            return e                                  # Baseline: doğrudan U-Net skip
        f_h = self.hfam[i](e, prev_mask, edge) if self.use_hfam else None
        f_p = self.pam[i](e, gabor) if self.use_pam else None
        if f_h is not None and f_p is not None:
            return self.fuse[i](torch.cat([f_h, f_p], dim=1))   # Denklem (10)
        return f_h if f_h is not None else f_p

    def forward(self, x):
        # Sabit ipuçları (Laplace farkı, 31×31 Gabor) karışık hassasiyette (bf16: 8 bit mantis) gürültülenmesin
        with torch.autocast(x.device.type, enabled=False):
            gray = rgb_to_grayscale(x.float())
            edge = self.edge_cue(gray) if self.need_edge else None
            gabor = self.gabor(gray) if self.use_pam else None

        e = [adj(f) for adj, f in zip(self.adjust, self.encoder(x))]
        e[4] = self.gc(e[4])

        d = self.dec5(e[4])                           # d5 (/16)
        masks = [self.mask_heads[4](d)]
        bounds = [self.boundary_heads[4](d)]
        for i in (3, 2, 1, 0):                        # seviye 4 → 1
            s = self.skip(i, e[i], masks[-1], edge, gabor)
            d = self.dec[i](s, d)
            masks.append(self.mask_heads[i](d))
            bounds.append(self.boundary_heads[i](d))

        # Tam çözünürlükten kabaya: [MP1, MP2, MP3, MP4, MP5]
        return masks[::-1], bounds[::-1]


# mimariyi belirleyen komut satırı seçenekleri (checkpoint'teki args'tan geri yüklenir); eksikse eski koşu = varsayılan
ARCH_DEFAULTS = {'backbone': 'res2net50', 'no_hfam': False, 'no_pam': False, 'n_gabor': 4, 'hfam_residual': False,
                 'no_dec_norm': True, 'edge_op': 'laplace', 'texture_op': 'gabor', 'hfam_no_bg': False,
                 'hfam_no_bdy': False, 'hfam_no_edge': False, 'no_attn_refine': False, 'gc_block': 'none'}


def glpoly_from_args(a, pretrained=False):
    """train.py'nin args sözlüğünden (checkpoint['args'] ya da vars(opt)) modeli kurar.

    no_dec_norm'un varsayılanı True: bu seçenekten önceki checkpoint'ler normalizasyonsuz kod çözücüyle eğitildi.
    """
    g = {k: a.get(k, v) for k, v in ARCH_DEFAULTS.items()}
    parts = tuple(p for p in PARTS if not g[f'hfam_no_{p}'])
    return GLPoly(g['backbone'], pretrained=pretrained, use_hfam=not g['no_hfam'], use_pam=not g['no_pam'],
                  n_gabor=g['n_gabor'], hfam_residual=g['hfam_residual'], dec_norm=not g['no_dec_norm'],
                  edge_op=g['edge_op'], texture_op=g['texture_op'], hfam_parts=parts,
                  attn_refine=not g['no_attn_refine'], gc_block=g['gc_block'])


def param_groups(model, backbone_lr_mult, weight_decay, stage_mults=None):
    """Backbone ×mult LR; weight decay yalnızca conv/linear ağırlıklarına (bias, BN/LN hariç).

    stage_mults: gövdede aşama başına çarpan (backbone_stage_of sırasıyla 4 değer); verilirse backbone_lr_mult
    yerine geçer. None iken gruplar ve sıraları eskisiyle aynı (eski last.pth'lerin optimizer durumu yüklenir)."""
    enc = model.encoder
    if stage_mults is not None and not hasattr(enc, 'backbone_stage_of'):
        raise ValueError('aşama bazlı kodlayıcı LR (--bb_stage_lr_mult) yalnızca convnext_tiny ile')
    stage_of = enc.backbone_stage_of() if stage_mults is not None else None
    bb_params = set(map(id, enc.backbone_parameters() if hasattr(enc, 'backbone_parameters')
                        else enc.parameters()))
    groups = {(b, d): [] for b in (True, False) for d in (True, False)}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        groups[(id(p) in bb_params, p.ndim > 1)].append(p)
    out = []
    for (is_bb, decay), params in groups.items():
        if is_bb and stage_of is not None:
            for s, mult in enumerate(stage_mults):
                ps = [p for p in params if stage_of[id(p)] == s]
                if ps:
                    out.append({'params': ps, 'lr_mult': mult, 'weight_decay': weight_decay if decay else 0.0})
        elif params:
            out.append({'params': params, 'lr_mult': backbone_lr_mult if is_bb else 1.0,
                        'weight_decay': weight_decay if decay else 0.0})
    return out
