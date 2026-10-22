"""Classifier using ResNet and APAG0-style extra branches."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from model.APAGRes import APAGRes
from model.layers import SumClassificationHead


class Network(nn.Module):
    def __init__(self, num_classes=1000, backbone_name='resnet18',
                 backbone_pretrained=False):
        super().__init__()
        self.apagres = APAGRes(
            backbone_name=backbone_name, pretrained=backbone_pretrained)
        self.classifier = SumClassificationHead(
            in_channels=self.apagres.out_channels,
            out_channels=num_classes,
        )

    def forward(self, x):
        return self.classifier(self.apagres(x))


if __name__ == '__main__':
    model = Network(backbone_name='resnet18').eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 256, 384))
    print('output.shape:', output.shape)
