
import torch
from torch import nn
from torch.nn import functional as F

class RConv1(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, bias=False):
        super(RConv1, self).__init__()

        self.inp = inp
        self.oup = oup

        padding = (kernel_size - 1) // 2

        layers = []
        layers.extend([
            nn.Conv2d(inp, oup, kernel_size, 1, padding, bias=bias),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv(x)

        if self.inp == self.oup:
            y = y + x

        y = F.relu(y)

        return y

class DSConv(nn.Module):
    def __init__(self, inp, outp, s=1):
        super(DSConv, self).__init__()
        self.inp = inp
        self.outp = outp

        self.conv_dw = nn.Conv2d(inp, inp, (3,3), groups=inp, stride=s, padding=1)
        self.conv_pt1 = nn.Conv2d(inp, inp, 1, stride=s, padding=0)
        self.conv_pt2 = nn.Conv2d(inp, outp, 1, stride=s, padding=0)
        self.bn1 = nn.BatchNorm2d(inp)
        self.bn2 = nn.BatchNorm2d(outp)

        if self.inp != self.outp:
            self.skip = nn.Conv2d(inp, outp,1, groups=inp, stride=s, padding=0)


    def forward(self,x):
        y = self.conv_dw(x)
        y = self.conv_pt1(y)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv_pt2(y)
        y = self.bn2(y)

        if self.inp != self.outp:
            y = y + self.skip(x)
        else:
            y = y + x

        return y

class SumClassificationHead(nn.Module):
    """Pool, activate and classify each input, then sum the logits.

    in_channels can be a sequence, with one channel count per input, or an
    integer repeated num_inputs times. The output shape is [batch, out_channels].
    Each input has its own classifier; weights are not shared across inputs.
    """

    def __init__(self, in_channels, out_channels, num_inputs=None):
        super().__init__()
        if num_inputs is not None and (type(num_inputs) is not int or num_inputs <= 0):
            raise ValueError('num_inputs must be a positive integer')
        if isinstance(in_channels, int):
            num_inputs = 1 if num_inputs is None else num_inputs
            channels = (in_channels,) * num_inputs
        else:
            channels = tuple(in_channels)
            if num_inputs is not None and num_inputs != len(channels):
                raise ValueError('num_inputs must match the length of in_channels')
        if not channels or any(type(value) is not int or value <= 0 for value in channels):
            raise ValueError('in_channels must contain positive integers')
        if type(out_channels) is not int or out_channels <= 0:
            raise ValueError('out_channels must be a positive integer')
        self.num_inputs = len(channels)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in, out_channels, 1, bias=True),
            )
            for channels_in in channels
        ])

    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)
        if len(inputs) != self.num_inputs:
            raise ValueError(f'Expected {self.num_inputs} inputs, got {len(inputs)}')
        batch_size = inputs[0].size(0)
        output = None
        for head, feature in zip(self.heads, inputs):
            if feature.ndim != 4 or feature.size(0) != batch_size:
                raise ValueError('Inputs must be NCHW tensors with the same batch size')
            logits = head(feature)
            output = logits if output is None else output + logits
        return output.flatten(1)

