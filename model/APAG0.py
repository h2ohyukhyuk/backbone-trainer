"""APAGNet0 backbone with its original layer names and feature ordering.

Classification networks inherit this module so state_dict keys stay at the root
(e.g. layer1.0.weight), preserving pre-refactor checkpoint compatibility.
"""
import torch.nn as nn
from model.layers import DSConv, RConv1 as ResidualConv1


class APAG0(nn.Module):
    def __init__(self):
        super().__init__()

        # 256 -> 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, stride=2),
            nn.Tanh(),
            DSConv(16, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        # 64 -> 32
        self.layer2 = nn.Sequential(
            DSConv(64, 64),
            nn.ReLU(inplace=True),
            DSConv(64, 64),
            nn.ReLU(inplace=True),
            DSConv(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer2_extra = nn.Sequential(
            ResidualConv1(64, 64, 1, False),
            ResidualConv1(64, 64, 1, False),
            DSConv(64, 128),
            nn.ReLU(inplace=True),
        )

        # 32 -> 16
        self.layer3 = nn.Sequential(
            DSConv(64, 128),
            nn.ReLU(inplace=True),
            DSConv(128, 128),
            nn.ReLU(inplace=True),
            DSConv(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer3_extra = nn.Sequential(
            ResidualConv1(128, 128, 1, False),
            ResidualConv1(128, 128, 1, False),
            DSConv(128, 128),
            nn.ReLU(inplace=True),
        )

        # 16 -> 8
        self.layer4 = nn.Sequential(
            DSConv(128, 256),
            nn.ReLU(inplace=True),
            DSConv(256, 256),
            nn.ReLU(inplace=True),
            DSConv(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer4_extra = nn.Sequential(
            ResidualConv1(256, 128, 1, False),
            ResidualConv1(128, 128, 1, False),
            DSConv(128, 128),
            nn.ReLU(inplace=True),
        )
        # Keep spatial resolution; layer5 does not downsample.
        self.layer5 = nn.Sequential(
            DSConv(256, 256),
            nn.ReLU(inplace=True),
            DSConv(256, 256),
            nn.ReLU(inplace=True),
        )

    def forward_features(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        c1 = self.layer2_extra(x2)
        x3 = self.layer3(x2)
        c2 = self.layer3_extra(x3)
        x4 = self.layer4(x3)
        c3 = self.layer4_extra(x4)
        c4 = self.layer5(x4)
        return c4, c1, c2, c3

    def forward(self, x):
        return self.forward_features(x)
