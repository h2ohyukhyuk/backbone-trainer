"""Multi-resolution APAG Transformer; fine/coarse stems meet at /4."""
import torch
from torch import nn
from torch.nn import functional as F
from model.layers import DSConv
from model.APAGTF import APAGTF


class APAGMRTF(APAGTF):
    def __init__(self, mlp_hidden=(117, 234, 475)):
        super().__init__(mlp_hidden=mlp_hidden)
        del self.layer1
        self.layer1_fine = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, stride=2), nn.ReLU(inplace=True),
            DSConv(16, 32), nn.MaxPool2d(2, 2), nn.ReLU(inplace=True))
        self.layer1_coarse = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(inplace=True),
            DSConv(8, 32), nn.ReLU(inplace=True))

    def stem(self, x):
        fine = self.layer1_fine(x)
        # Match the fine grid exactly, including non-multiple-of-four inputs.
        coarse = F.interpolate(x, size=fine.shape[-2:], mode='bilinear', align_corners=False)
        coarse = self.layer1_coarse(coarse)
        return torch.cat((fine, coarse), dim=1)
