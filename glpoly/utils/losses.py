"""Kayıp fonksiyonları, makale §3.4, Denklem (13)–(16)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def structure_loss(logits, mask):
    """PraNet structure loss: ağırlıklı BCE + ağırlıklı IoU (W_map = sınıra yakın piksellere ağırlık)."""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(logits)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def bce_dice_loss(logits, target, smooth=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + smooth) / (prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
    return bce + dice.mean()


class GLPolyLoss(nn.Module):
    """L_total = Σ w_i·Structure(MP_i, Y_i^m) + λ_b · Σ w'_i·BCEDice(BP_i, Y_i^b)."""

    def __init__(self, mask_weights=(1.0, 0.5, 0.3, 0.2, 0.15), boundary_weights=None, lambda_b=0.8):
        super().__init__()
        self.w = list(mask_weights)
        self.wb = list(boundary_weights) if boundary_weights else self.w
        self.lambda_b = lambda_b

    def forward(self, masks, bounds, gt, gt_boundary):
        l_mask = l_bdy = 0.0
        for w, wb, mp, bp in zip(self.w, self.wb, masks, bounds):
            y = F.interpolate(gt, size=mp.shape[2:], mode='bilinear', align_corners=False) \
                if mp.shape[2:] != gt.shape[2:] else gt
            yb = F.interpolate(gt_boundary, size=bp.shape[2:], mode='bilinear', align_corners=False) \
                if bp.shape[2:] != gt_boundary.shape[2:] else gt_boundary
            l_mask = l_mask + w * structure_loss(mp, y)
            l_bdy = l_bdy + wb * bce_dice_loss(bp, yb)
        total = l_mask + self.lambda_b * l_bdy
        return total, l_mask.detach(), l_bdy.detach()
