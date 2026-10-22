"""ResNet feature backbone with APAG0-style post-downsampling branches."""
import torch.nn as nn
import timm
from timm.models import FeatureInfo

from model.layers import DSConv, RConv1


class _ResNetFeatures(nn.Module):
    """Expose /8, /16, early /32 and final /32 without adding any blocks."""

    def __init__(self, backbone):
        super().__init__()
        # Retain timm's feature module names and pretrained parameters, excluding
        # the classification pooling and FC layers.
        for name in ('conv1', 'bn1', 'act1', 'maxpool',
                     'layer1', 'layer2', 'layer3', 'layer4'):
            setattr(self, name, getattr(backbone, name))
        feature_info = [dict(info) for info in backbone.feature_info[2:]]
        early = dict(feature_info[-1], module='layer4.0')
        feature_info.insert(-1, early)
        self.feature_info = FeatureInfo(feature_info, out_indices=(0, 1, 2, 3))

    def forward(self, x):
        x = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4[0](x2)
        x4 = x3
        for block in self.layer4[1:]:
            x4 = block(x4)
        return x1, x2, x3, x4


class APAGRes(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=False):
        super().__init__()
        if backbone_name not in ('resnet18', 'resnet50', 'resnet101'):
            raise ValueError('backbone_name must be resnet18, resnet50 or resnet101')
        self.backbone_name = backbone_name
        backbone = timm.create_model(backbone_name, pretrained=pretrained)
        self.backbone = _ResNetFeatures(backbone)
        c1, c2, c3, c4 = self.backbone.feature_info.channels()
        self.out_channels = (c4, 128, 128, 128)

        # Adapt channels before DSConv, whose grouped skip needs divisibility.
        self.layer2_extra = nn.Sequential(
            RConv1(c1, 128, 1, False),
            RConv1(128, 128, 1, False),
            DSConv(128, 128),
            nn.ReLU(inplace=True),
        )
        self.layer3_extra = nn.Sequential(
            RConv1(c2, 128, 1, False),
            RConv1(128, 128, 1, False),
            DSConv(128, 128),
            nn.ReLU(inplace=True),
        )
        self.layer4_extra = nn.Sequential(
            RConv1(c3, 128, 1, False),
            RConv1(128, 128, 1, False),
            DSConv(128, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        e2 = self.layer2_extra(x1)
        e3 = self.layer3_extra(x2)
        e4 = self.layer4_extra(x3)
        return x4, e2, e3, e4
