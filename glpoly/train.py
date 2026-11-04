"""GLPoly eğitimi (makale §4.3): AdamW, backbone ×0.33 LR, warm-up + cosine, çok ölçekli eğitim,
derin denetim, doğrulamada eşik taraması.

Örnek:
    python train.py --backbone res2net50 --name glpoly_res2net
    python train.py --backbone convnext_tiny --no_pam --name abl_hfam   # ablation
"""
import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.glpoly import ARCH_DEFAULTS, glpoly_from_args, param_groups
from utils.data import PolypEval, PolypTrain, list_pairs, split_train_val
from utils.losses import GLPolyLoss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True, help='runs/<name> altına kaydedilir')
    p.add_argument('--backbone', default='res2net50', choices=['res2net50', 'convnext_tiny'])
    p.add_argument('--no_hfam', action='store_true')
    p.add_argument('--no_pam', action='store_true')
    p.add_argument('--n_gabor', type=int, default=4)
    p.add_argument('--hfam_residual', action='store_true', help='HFAM\'e makalede olmayan + E_i ekle (ek deney)')
    p.add_argument('--no_dec_norm', action='store_true',
                   help='kod çözücüdeki ConvT sonrası BN+ReLU\'yu kaldır (makaledeki/orj yapısı; aktivasyonlar büyür)')
    # ablation'lar ve kontroller (varsayılanlar = makaledeki model)
    p.add_argument('--edge_op', default='laplace', choices=['laplace', 'sobel', 'learn'],
                   help='HFAM görüntü kenarı: Laplace (makale), Sobel ya da öğrenilebilir 5×5 conv')
    p.add_argument('--texture_op', default='gabor', choices=['gabor', 'learn'],
                   help='PAM doku girdisi: sabit Gabor (makale) ya da aynı boyutta öğrenilebilir conv')
    p.add_argument('--hfam_no_bg', action='store_true', help='HFAM: ters dikkat dalı yok')
    p.add_argument('--hfam_no_bdy', action='store_true', help='HFAM: maske Laplace (sınır) dalı yok')
    p.add_argument('--hfam_no_edge', action='store_true', help='HFAM: görüntü kenarı dalı yok')
    p.add_argument('--no_attn_refine', action='store_true', help='HFAM\'de ECA+SA, PAM\'de ECA yok')
    p.add_argument('--color_aug', default='none', choices=['none', 'jitter', 'lab'],
                   help='renk artırması: yok (makale), ColorJitter ya da SANet tarzı LAB renk değişimi')
    p.add_argument('--gc_block', default='none', choices=['none', 'attn'],
                   help='/32\'de genel bağlam bloğu (transformer katmanı; darboğaz deneyi, makalede yok)')
    p.add_argument('--clahe_p', type=float, default=0.0, help='rastgele CLAHE olasılığı (0 = makale)')
    p.add_argument('--mask_corrupt', type=float, default=0.0,
                   help='eğitimde HFAM kaba maskesini bozma olasılığı (aşındırma/genişletme; 0 = makale)')
    p.add_argument('--train_root', default=os.path.join(ROOT, 'dataset/TrainDataset'))
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--max_train_images', type=int, default=0, help='duman testi için (0 = hepsi)')
    p.add_argument('--trainsize', type=int, default=352)
    p.add_argument('--size_rates', type=float, nargs='+', default=[0.75, 1.0, 1.25])
    p.add_argument('--epoch', type=int, default=100)
    p.add_argument('--batchsize', type=int, default=8)
    p.add_argument('--accum', type=int, default=1, help='gradyan biriktirme adımı')
    p.add_argument('--lr', type=float, default=6e-4)
    p.add_argument('--bb_lr_mult', type=float, default=0.33)
    p.add_argument('--bb_stage_lr_mult', type=float, nargs=4, default=None,
                   help='ConvNeXt gövdesinde aşama başına LR çarpanı (/4, /8, /16, /32); bb_lr_mult yerine')
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--eta_min_ratio', type=float, default=1e-3, help='cosine sonu LR / base LR')
    p.add_argument('--ds_weights', type=float, nargs=5, default=[1.0, 0.5, 0.3, 0.2, 0.15])
    p.add_argument('--lambda_b', type=float, default=0.8)
    p.add_argument('--grad_clip', type=float, default=0.5)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split_seed', type=int, default=None,
                   help='doğrulama ayrımının seed\'i (verilmezse --seed); ek seed koşularında ayrım sabit tutulur')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else
                   ('mps' if torch.backends.mps.is_available() else 'cpu'))
    p.add_argument('--amp', action='store_true', help='CUDA: karışık hassasiyet (autocast)')
    p.add_argument('--amp_dtype', default='bf16', choices=['bf16', 'fp16'],
                   help='bf16: fp32 ile aynı aralık, GradScaler gerekmez. fp16 + GradScaler 26 Ekim E3 koşusunda '
                        'epoch 46\'da NaN\'a gitti')
    p.add_argument('--max_nonfinite', type=int, default=20,
                   help='bir epoch\'ta bu kadar sonlu olmayan kayıp olursa eğitimi durdur')
    p.add_argument('--resume', action='store_true', help='runs/<name>/last.pth üzerinden devam et')
    return p.parse_args(argv)


def save_atomic(obj, path):
    """Yarıda kesilen yazma bozuk checkpoint bırakmasın: önce geçici dosya, sonra yer değiştir."""
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_factor(step, warmup_steps, total_steps, eta_min_ratio):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def validate(model, loader, device, thresholds):
    """Her görüntü için eşik taraması; ortalama Dice'ı en yüksek eşik seçilir."""
    model.eval()
    dice_sum = torch.zeros(len(thresholds))
    n = 0
    for img, gt, _ in loader:
        masks, _ = model(img.to(device))
        prob = torch.sigmoid(F.interpolate(masks[0], size=gt.shape[-2:], mode='bilinear',
                                           align_corners=False)).cpu()[:, 0]
        for b in range(prob.shape[0]):
            g = gt[b]
            for j, th in enumerate(thresholds):
                pb = (prob[b] >= th).float()
                inter = (pb * g).sum()
                dice_sum[j] += (2 * inter + 1e-8) / (pb.sum() + g.sum() + 1e-8)
            n += 1
    dice = dice_sum / n
    best = int(dice.argmax())
    model.train()
    return dice[best].item(), float(thresholds[best]), dice[len(thresholds) // 2].item()


def main(argv=None):
    opt = get_args(argv)
    seed_all(opt.seed)
    run_dir = os.path.join(ROOT, 'runs', opt.name)
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device(opt.device)
    use_amp = opt.amp and device.type == 'cuda'
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    amp_dtype = torch.bfloat16 if opt.amp_dtype == 'bf16' else torch.float16
    # bf16'nın aralığı fp32 ile aynı; ölçekleyici yalnızca fp16'da gerekir
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)

    pairs = list_pairs(opt.train_root)
    train_pairs, val_pairs = split_train_val(pairs, opt.val_ratio,
                                             opt.seed if opt.split_seed is None else opt.split_seed)
    if opt.max_train_images:
        train_pairs = train_pairs[:opt.max_train_images]
        val_pairs = val_pairs[:max(8, opt.max_train_images // 10)]
    with open(os.path.join(run_dir, 'split.json'), 'w') as f:
        json.dump({'train': [os.path.basename(a) for a, _ in train_pairs],
                   'val': [os.path.basename(a) for a, _ in val_pairs]}, f)

    train_loader = DataLoader(PolypTrain(train_pairs, opt.trainsize, color_aug=opt.color_aug, clahe_p=opt.clahe_p),
                              batch_size=opt.batchsize,
                              shuffle=True, num_workers=opt.workers, drop_last=True,
                              persistent_workers=opt.workers > 0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(PolypEval(val_pairs, opt.trainsize), batch_size=1, num_workers=2,
                            persistent_workers=True)

    last_path, best_path = os.path.join(run_dir, 'last.pth'), os.path.join(run_dir, 'best.pth')
    ck = None
    if opt.resume:
        if not os.path.exists(last_path):
            raise FileNotFoundError(f'--resume verildi ama {last_path} yok')
        ck = torch.load(last_path, map_location='cpu', weights_only=False)
        # mimari, checkpoint'i yazan koşudan gelsin (dec_norm seçeneğinden önceki koşularda yoktu = kapalı)
        for k, v in ARCH_DEFAULTS.items():
            setattr(opt, k, ck['args'].get(k, v))

    model = glpoly_from_args(vars(opt), pretrained=True).to(device)
    for h in model.hfam or []:
        h.corrupt = opt.mask_corrupt
    groups = param_groups(model, opt.bb_lr_mult, opt.weight_decay, opt.bb_stage_lr_mult)
    for g in groups:
        g['lr'] = opt.lr * g['lr_mult']
    optimizer = torch.optim.AdamW(groups, lr=opt.lr)
    criterion = GLPolyLoss(opt.ds_weights, lambda_b=opt.lambda_b)

    # epoch sonunda yarım kalan biriktirme de bir adım sayılır
    steps_per_epoch = math.ceil(len(train_loader) * len(opt.size_rates) / opt.accum)
    total_steps = steps_per_epoch * opt.epoch
    warmup_steps = steps_per_epoch * opt.warmup_epochs
    thresholds = np.round(np.arange(0.05, 0.951, 0.05), 2)

    start_epoch, global_step, best_dice = 1, 0, -1.0
    if ck is not None:
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        if 'scaler' in ck:
            scaler.load_state_dict(ck['scaler'])
        start_epoch, global_step, best_dice = ck['epoch'] + 1, ck['step'], ck['best_dice']
        print(f'devam: epoch {start_epoch}, best val dice {best_dice:.4f}')

    # devamda ilk koşunun ayarları ezilmesin
    args_name = f'args_resume_{start_epoch}.json' if opt.resume else 'args.json'
    with open(os.path.join(run_dir, args_name), 'w', encoding='utf-8') as f:
        json.dump(vars(opt), f, indent=2)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log = open(os.path.join(run_dir, 'log.txt'), 'a', encoding='utf-8')

    def say(msg):
        line = f'{datetime.now():%Y-%m-%d %H:%M:%S} {msg}'
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()

    say(f'model={opt.backbone} hfam={not opt.no_hfam} pam={not opt.no_pam} edge={opt.edge_op} '
        f'texture={opt.texture_op} hfam_parts={[p for p in ("bg", "bdy", "edge") if not getattr(opt, "hfam_no_" + p)]} '
        f'attn_refine={not opt.no_attn_refine} color={opt.color_aug} clahe_p={opt.clahe_p} '
        f'mask_corrupt={opt.mask_corrupt} bb_lr_mult={opt.bb_lr_mult} bb_stage={opt.bb_stage_lr_mult} gc={opt.gc_block} params={n_params:.2f}M '
        f'train={len(train_pairs)} val={len(val_pairs)} device={device}')
    writer = SummaryWriter(os.path.join(run_dir, 'tb'))

    head_group = next(g for g in optimizer.param_groups if g['lr_mult'] == 1.0)

    def optimizer_step():
        """Bir optimizer adımı. Gradyanlar sonlu değilse adımı atlar ve False döner (LR sayacı yine ilerler)."""
        nonlocal global_step
        for g in optimizer.param_groups:
            g['lr'] = opt.lr * g['lr_mult'] * lr_factor(global_step, warmup_steps,
                                                        total_steps, opt.eta_min_ratio)
        scaler.unscale_(optimizer)
        grads = [p.grad for g in optimizer.param_groups for p in g['params'] if p.grad is not None]
        # GradScaler yalnızca fp16'da kendi kontrolünü yapar; bf16/fp32'de NaN gradyan ağırlıklara geçmesin
        # hızlı yol: L2 normları (NaN/inf yayılır); sonlu ama çok büyük gradyanda norm taşabilir, o zaman tek tek bak
        ok = scaler.is_enabled() or bool(torch.isfinite(torch.stack(torch._foreach_norm(grads))).all()) \
            or all(bool(torch.isfinite(g).all()) for g in grads)
        if ok:
            if opt.grad_clip:
                torch._foreach_clamp_min_(grads, -opt.grad_clip)
                torch._foreach_clamp_max_(grads, opt.grad_clip)
            scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        return ok

    def abort(msg):
        """NaN korumaları: çıkış kodu 3 (kuyruk bu durumda fp32 ile devam eder); last.pth önceki epoch'ta kalır."""
        say(msg)
        log.close()
        sys.exit(3)

    model.train()
    for epoch in range(start_epoch, opt.epoch + 1):
        t0 = time.time()
        sums = np.zeros(3)
        n_iter = n_bad = 0
        optimizer.zero_grad(set_to_none=True)
        for i, (img, gt, bdy) in enumerate(train_loader, 1):
            for rate in opt.size_rates:
                size = int(round(opt.trainsize * rate / 32) * 32)
                x, y, yb = img.to(device), gt.to(device), bdy.to(device)
                if size != opt.trainsize:
                    x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
                    y = F.interpolate(y, size=(size, size), mode='bilinear', align_corners=False)
                    yb = F.interpolate(yb, size=(size, size), mode='bilinear', align_corners=False)
                with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    masks, bounds = model(x)
                masks, bounds = [m.float() for m in masks], [b.float() for b in bounds]
                loss, l_m, l_b = criterion(masks, bounds, y, yb)
                lv = loss.item()
                if not math.isfinite(lv):
                    # bu batch'i atla; tekrarlarsa sessizce NaN'la devam etmek yerine dur
                    n_bad += 1
                    optimizer.zero_grad(set_to_none=True)
                    say(f'UYARI epoch {epoch} batch {i} ölçek {rate}: kayıp sonlu değil ({n_bad}/{opt.max_nonfinite})')
                    # ileri geçişteki NaN BN istatistiklerini bozmuşsa epoch'un kalanı boşa gider: hemen dur
                    bad_bufs = [n for n, b in model.named_buffers()
                                if b.is_floating_point() and not torch.isfinite(b).all()]
                    if bad_bufs:
                        abort(f'HATA: ileri geçiş BN istatistiklerini bozdu ({bad_bufs[:3]}); last.pth önceki epoch\'ta')
                    if n_bad >= opt.max_nonfinite:
                        abort('HATA: çok fazla sonlu olmayan kayıp; last.pth bir önceki epoch\'ta kaldı')
                    continue
                scaler.scale(loss / opt.accum).backward()
                n_iter += 1
                sums += [lv, l_m.item(), l_b.item()]
                if n_iter % opt.accum == 0 and not optimizer_step():
                    n_bad += 1
                    say(f'UYARI epoch {epoch} batch {i} ölçek {rate}: gradyan sonlu değil, adım atlandı '
                        f'({n_bad}/{opt.max_nonfinite})')
                    if n_bad >= opt.max_nonfinite:
                        abort('HATA: çok fazla sonlu olmayan gradyan; last.pth bir önceki epoch\'ta kaldı')
            if i % 20 == 0 or i == len(train_loader):
                avg = sums / max(1, n_iter)
                say(f'epoch {epoch:03d}/{opt.epoch} step {i:04d}/{len(train_loader)} '
                    f'loss {avg[0]:.4f} (mask {avg[1]:.4f}, bdy {avg[2]:.4f}) '
                    f'lr {head_group["lr"]:.2e} {(time.time() - t0) / i:.2f}s/batch')
        if n_iter % opt.accum:
            optimizer_step()

        train_time = time.time() - t0
        broken = [n for n, t in [*model.named_parameters(), *model.named_buffers()]
                  if t.is_floating_point() and not torch.isfinite(t).all()]
        if broken:
            abort(f'HATA epoch {epoch}: {len(broken)} tensörde NaN/inf (ör. {broken[:3]}); last.pth korunuyor')
        val_dice, val_th, val_dice05 = validate(model, val_loader, device, thresholds)
        avg = sums / max(1, n_iter)
        for k, v in zip(('loss', 'mask', 'boundary'), avg):
            writer.add_scalar(f'train/{k}', v, epoch)
        writer.add_scalar('val/dice_best_th', val_dice, epoch)
        writer.add_scalar('val/dice_0.5', val_dice05, epoch)
        writer.add_scalar('val/threshold', val_th, epoch)

        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                 'step': global_step, 'best_dice': best_dice, 'threshold': val_th, 'args': vars(opt),
                 'scaler': scaler.state_dict()}
        if val_dice > best_dice:
            best_dice = state['best_dice'] = val_dice
            save_atomic({k: state[k] for k in ('model', 'epoch', 'threshold', 'args', 'best_dice')}, best_path)
        save_atomic(state, last_path)
        say(f'EPOCH {epoch:03d} loss {avg[0]:.4f} val_dice {val_dice:.4f}@{val_th:.2f} '
            f'(0.5: {val_dice05:.4f}) best {best_dice:.4f} train {train_time / 60:.1f} dk '
            f'toplam {(time.time() - t0) / 60:.1f} dk' + (f' atlanan {n_bad}' if n_bad else ''))
    writer.close()
    log.close()


if __name__ == '__main__':
    main()
