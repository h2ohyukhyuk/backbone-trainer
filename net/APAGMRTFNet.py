"""APAGMRTF ImageNet classifier with average-pooled, summed stage logits."""
from pathlib import Path
import sys

# Support both direct execution and importing via the project's model loader.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from model.APAGMRTF import APAGMRTF
from model.APAGTF import profile_macs
from model.layers import SumClassificationHead


class Network(nn.Module):
    def __init__(self, num_classes=1000, mlp_hidden=(117, 234, 475)):
        super().__init__()
        self.apagmrtf = APAGMRTF(mlp_hidden=mlp_hidden)
        self.classifier = SumClassificationHead(
            in_channels=(256, 128, 128), out_channels=num_classes)

    def forward(self, x):
        return self.classifier(self.apagmrtf(x))


if __name__ == '__main__':
    torch.set_num_threads(2)
    model = Network().eval()
    shape = (1, 3, 256, 384)
    with torch.no_grad():
        print('output.shape:', tuple(model(torch.zeros(shape)).shape))
    macs, params = profile_macs(model, shape)
    print(f'Params: {params / 1e6:.3f}M | MACs: {macs / 1e6:.3f}M '
          f'| FLOPs (2 per MAC): {2 * macs / 1e6:.3f}M')
