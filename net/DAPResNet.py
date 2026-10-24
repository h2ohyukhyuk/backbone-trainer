"""DAPRes classifier with global average pooling and a linear head."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from model.DAPRes import DAPRes


class Network(nn.Module):
    def __init__(self, num_classes=1000, backbone_name='resnet18',
                 backbone_pretrained=False):
        super().__init__()
        self.dapres = DAPRes(
            backbone_name=backbone_name, pretrained=backbone_pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.dapres.out_channels, num_classes)

    def forward(self, x):
        x = self.dapres(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


if __name__ == '__main__':
    model = Network(backbone_name='resnet18').eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 256, 384))
    print('output.shape:', output.shape)
