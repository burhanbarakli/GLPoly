"""Veri yükleyiciler (BB-Res2Net_orj/utils/dataloader.py'den uyarlandı).

Eğitim/doğrulama ayrımı TrainDataset içinden sabit seed ile yapılır; test setleri yalnızca
nihai değerlendirmede kullanılır.
"""
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def list_pairs(root):
    imgs = sorted(f for f in os.listdir(os.path.join(root, 'images')) if f.lower().endswith(IMG_EXT))
    gts = sorted(f for f in os.listdir(os.path.join(root, 'masks')) if f.lower().endswith(IMG_EXT))
    assert len(imgs) == len(gts), root
    for a, b in zip(imgs, gts):
        assert os.path.splitext(a)[0] == os.path.splitext(b)[0], (a, b)
    return [(os.path.join(root, 'images', a), os.path.join(root, 'masks', b)) for a, b in zip(imgs, gts)]


def split_train_val(pairs, val_ratio, seed):
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    n_val = int(round(len(pairs) * val_ratio))
    val = sorted(idx[:n_val])
    train = sorted(idx[n_val:])
    return [pairs[i] for i in train], [pairs[i] for i in val]


def mask_to_boundary(mask_np, k=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.dilate(mask_np, kernel) - cv2.erode(mask_np, kernel)


def lab_exchange(img, ref):
    """SANet'teki renk değişimi: görüntünün LAB kanal ortalama/std'sini rastgele bir eğitim görüntüsününkine eşitler."""
    a = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(np.asarray(ref), cv2.COLOR_RGB2LAB).astype(np.float32)
    am, asd = a.mean((0, 1)), a.std((0, 1)) + 1e-6
    bm, bsd = b.mean((0, 1)), b.std((0, 1))
    out = np.clip((a - am) / asd * bsd + bm, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_LAB2RGB))


def clahe(img, clip):
    """L kanalında CLAHE (8×8 karo): yerel kontrastı artırır (revizyon/darbogaz_analizi.md, D2)."""
    lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


class PolypTrain(Dataset):
    """color_aug: 'none' (makale), 'jitter' (p=0.8 parlaklık/kontrast/doygunluk/ton) ya da 'lab' (SANet renk değişimi).
    clahe_p: bu olasılıkla rastgele CLAHE (clip 1–3); 0 = makale."""

    def __init__(self, pairs, size, augment=True, color_aug='none', clahe_p=0.0):
        self.pairs, self.size, self.augment, self.color_aug = pairs, size, augment, color_aug
        self.clahe_p = clahe_p
        self.norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
        self.jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.02)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img = Image.open(self.pairs[i][0]).convert('RGB').resize((self.size, self.size), Image.BILINEAR)
        gt = Image.open(self.pairs[i][1]).convert('L').resize((self.size, self.size), Image.NEAREST)
        if self.augment:
            # renk, döndürmeden önce: döndürmenin siyah köşeleri renk istatistiğine girmesin ('none' RNG sırası aynı)
            if self.color_aug == 'jitter' and random.random() < 0.8:
                img = self.jitter(img)
            elif self.color_aug == 'lab':
                ref = Image.open(self.pairs[random.randrange(len(self.pairs))][0]).convert('RGB')
                img = lab_exchange(img, ref.resize((self.size, self.size), Image.BILINEAR))
            if self.clahe_p > 0 and random.random() < self.clahe_p:
                img = clahe(img, random.uniform(1.0, 3.0))
            if random.random() < 0.5:
                angle = random.uniform(-90, 90)
                img, gt = TF.rotate(img, angle), TF.rotate(gt, angle)
            if random.random() < 0.5:
                img, gt = TF.vflip(img), TF.vflip(gt)
            if random.random() < 0.5:
                img, gt = TF.hflip(img), TF.hflip(gt)
        gt_np = (np.array(gt) > 127).astype(np.uint8)
        boundary = mask_to_boundary(gt_np)
        return (self.norm(img),
                torch.from_numpy(gt_np).float().unsqueeze(0),
                torch.from_numpy(boundary).float().unsqueeze(0))


class PolypEval(Dataset):
    """Doğrulama/test: görüntü yeniden boyutlanır, GT orijinal boyutunda döner."""

    def __init__(self, pairs, size):
        self.pairs, self.size = pairs, size
        self.tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, gt_path = self.pairs[i]
        img = self.tf(Image.open(img_path).convert('RGB'))
        gt = (np.array(Image.open(gt_path).convert('L')) > 127).astype(np.float32)
        name = os.path.splitext(os.path.basename(img_path))[0] + '.png'
        return img, torch.from_numpy(gt), name
