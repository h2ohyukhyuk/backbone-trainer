"""APAGRes features fused by the lightweight APAG face-detector FPN."""
import torch.nn as nn

from model.APAGRes import APAGRes


class DWConv(nn.Module):
    """Depthwise/pointwise smoothing from apag_fpn_net2.py."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FPN(nn.Module):
    """Fuse /8, /16, early /32 and final /32 features into /8.

    Match net/apag_fpn_net2.py: lateral 1x1 projections, addition at
    /32, then two bilinear 2x upsampling steps with DWConv smoothing.
    Only the final backbone input channel count varies with the ResNet.
    """

    def __init__(self, backbone_channels, fpn_channels=128):
        super().__init__()
        self.lat4 = nn.Conv2d(backbone_channels, fpn_channels, 1)
        self.lat3 = nn.Conv2d(128, fpn_channels, 1)
        self.lat2 = nn.Conv2d(128, fpn_channels, 1)
        self.lat1 = nn.Conv2d(128, fpn_channels, 1)
        self.smooth3 = DWConv(fpn_channels)
        self.smooth2 = DWConv(fpn_channels)
        self.smooth1 = DWConv(fpn_channels)
        self.up = nn.Upsample(scale_factor=2.0, mode='bilinear',
                              align_corners=False)

    def forward(self, c1, c2, c3, c4):
        p4 = self.lat4(c4)
        p3 = self.smooth3(self.lat3(c3) + p4)
        p2 = self.smooth2(self.lat2(c2) + self.up(p3))
        p1 = self.smooth1(self.lat1(c1) + self.up(p2))
        return p1


class DAPRes(APAGRes):
    """Keep all APAGRes branches and return their fused 128-channel map.

    Use input height and width divisible by 32 for the FPN's fixed 2x
    upsampling, matching the reference implementation.
    """

    def __init__(self, backbone_name='resnet18', pretrained=False):
        super().__init__(backbone_name=backbone_name, pretrained=pretrained)
        self.fpn = FPN(backbone_channels=self.out_channels[0])
        self.out_channels = 128

    def forward(self, x):
        x4, e2, e3, e4 = super().forward(x)
        return self.fpn(e2, e3, e4, x4)
