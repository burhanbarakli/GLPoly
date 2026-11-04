"""Backbone'lar: her biri 5 seviye (/2, /4, /8, /16, /32) ham öznitelik E_f1..E_f5 döndürür."""
import torch.nn as nn

from .res2net import res2net50_v1b_26w_4s


class Res2NetEncoder(nn.Module):
    channels = [64, 256, 512, 1024, 2048]

    def __init__(self, pretrained=True):
        super().__init__()
        net = res2net50_v1b_26w_4s(pretrained=pretrained)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.maxpool = net.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4

    def forward(self, x):
        e1 = self.stem(x)                       # /2
        e2 = self.layer1(self.maxpool(e1))      # /4
        e3 = self.layer2(e2)                    # /8
        e4 = self.layer3(e3)                    # /16
        e5 = self.layer4(e4)                    # /32
        return [e1, e2, e3, e4, e5]


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-T ilk özniteliğini /4'te verir; /2 seviyesi (E1) için hafif bir stem eklenir.

    Bu stem makalede açıkça tanımlı değildir (PLAN.md §2, madde 1).
    """
    channels = [64, 96, 192, 384, 768]

    def __init__(self, pretrained=True, name='convnext_tiny.fb_in22k'):
        super().__init__()
        import timm
        self.stem2 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.body = timm.create_model(name, features_only=True, pretrained=pretrained,
                                      out_indices=(0, 1, 2, 3), drop_path_rate=0.1)

    def forward(self, x):
        return [self.stem2(x), *self.body(x)]

    def backbone_parameters(self):
        # ×0.33 LR yalnızca önceden eğitilmiş gövdeye uygulanır; yeni stem head gibi eğitilir
        return self.body.parameters()

    def backbone_stage_of(self):
        """Gövde parametresi id → aşama: 0 = stem + stages_0 (/4), 1 = /8, 2 = /16, 3 = /32 (aşama bazlı LR için)."""
        return {id(p): 0 if n.startswith('stem') else int(n.split('.')[0].split('_')[1])
                for n, p in self.body.named_parameters()}


def build_backbone(name, pretrained=True):
    if name == 'res2net50':
        return Res2NetEncoder(pretrained)
    if name == 'convnext_tiny':
        return ConvNeXtEncoder(pretrained)
    raise ValueError(f'bilinmeyen backbone: {name}')
