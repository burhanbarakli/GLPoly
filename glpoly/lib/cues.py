"""Tamamlayıcı ipuçları: Laplace piramidi (L) ve Gabor doku öznitelikleri (G).

Laplace kodu BB-Res2Net_orj/lib/modules.py'den alınmıştır (MEGANet).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gauss_kernel(channels=1, device='cpu'):
    kernel = torch.tensor([[1., 4., 6., 4., 1.],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    return kernel.repeat(channels, 1, 1, 1).to(device)


def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    return F.conv2d(img, kernel, groups=img.shape[1])


def downsample(x):
    return x[:, :, ::2, ::2]


def upsample(x, channels):
    # Sıfır ekleyerek 2x büyütme + Gauss yumuşatma
    b, c, h, w = x.shape
    up = torch.zeros(b, c, h * 2, w * 2, device=x.device, dtype=x.dtype)
    up[:, :, ::2, ::2] = x
    return conv_gauss(up, 4 * gauss_kernel(channels, device=x.device))


def _laplace_step(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels, device=img.device))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2:] != img.shape[2:]:
        up = F.interpolate(up, size=img.shape[2:])
    return img - up, down


def make_laplace(img, channels):
    """Tek seviyeli Laplace (makaledeki Δ(·) operatörü)."""
    return _laplace_step(img, channels)[0]


def make_laplace_pyramid(img, level, channels):
    """Denklem (2): L_k = G_k − UpSample(G_{k+1})."""
    pyr, current = [], img
    for _ in range(level):
        diff, current = _laplace_step(current, channels)
        pyr.append(diff)
    pyr.append(current)
    return pyr


SOBEL = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])


class EdgeCue(nn.Module):
    """HFAM'in görüntü kenarı girdisi L (1 × H/2 × W/2).

    laplace: makaledeki Laplace piramidi seviyesi L_level (varsayılan, level=1 → H/2)
    sobel:   aynı yumuşatılmış G_1 (H/2) üzerinde Sobel gradyan büyüklüğü (alternatif kenar operatörü)
    learn:   öğrenilebilir 5×5, stride 2 conv (Laplace adımındaki 5×5 Gauss çekirdeğiyle aynı boyut; parametre eşit kontrol)
    """

    def __init__(self, op='laplace', level=1):
        super().__init__()
        self.op, self.level = op, level
        if op == 'learn':
            self.conv = nn.Conv2d(1, 1, 5, stride=2, padding=2, bias=False)
        elif op == 'sobel':
            self.register_buffer('kernel', torch.stack([SOBEL, SOBEL.t()]).unsqueeze(1))
        elif op != 'laplace':
            raise ValueError(op)

    def forward(self, gray):
        if self.op == 'laplace':
            return make_laplace_pyramid(gray, 5, 1)[self.level]
        if self.op == 'learn':
            return self.conv(gray)
        g1 = downsample(conv_gauss(gray, gauss_kernel(1, device=gray.device)))
        g = F.conv2d(F.pad(g1, (1, 1, 1, 1), mode='reflect'), self.kernel)
        return torch.sqrt((g ** 2).sum(1, keepdim=True) + 1e-12)


class LearnableTexture(nn.Module):
    """Gabor bankasının yerine aynı boyutta (K × 31×31) öğrenilebilir conv (parametre eşit kontrol)."""

    def __init__(self, ksize=31, n_filters=4):
        super().__init__()
        self.conv = nn.Conv2d(1, n_filters, ksize, padding=ksize // 2, bias=False)

    def forward(self, x):
        return self.conv(x)


class GaborConv(nn.Module):
    """Sabit (öğrenilmeyen) Gabor filtre bankası, Denklem (3). K = n_filters yön."""

    def __init__(self, ksize=31, sigma=4.0, lambd=10.0, gamma=0.5, psi=0.0, n_filters=4):
        super().__init__()
        self.n_filters = n_filters
        half = ksize // 2
        y, x = torch.meshgrid(torch.linspace(-half, half, ksize),
                              torch.linspace(-half, half, ksize), indexing='ij')
        kernels = []
        for theta in np.arange(0, np.pi, np.pi / n_filters):
            theta = torch.tensor(theta, dtype=torch.float32)
            x_t = x * torch.cos(theta) + y * torch.sin(theta)
            y_t = -x * torch.sin(theta) + y * torch.cos(theta)
            gb = torch.exp(-0.5 * (x_t ** 2 + gamma ** 2 * y_t ** 2) / sigma ** 2) * \
                torch.cos(2 * np.pi * x_t / lambd + psi)
            kernels.append(gb)
        self.register_buffer('kernel', torch.stack(kernels).unsqueeze(1))

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding='same')
