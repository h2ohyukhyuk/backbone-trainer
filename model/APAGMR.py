import torch
import torch.nn as nn
from model.layers import DSConv, RConv1


class APAGMR(nn.Module):
    def __init__(self):
        super().__init__()

        # 256 -> 64
        self.layer1_fine = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, stride=2),
                                    nn.ReLU(inplace=True),
                                    DSConv(16, 32),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True))

        self.layer1_coarse = nn.Sequential(nn.Upsample(scale_factor=0.25, mode="bilinear", align_corners=False),
                                    nn.Conv2d(3, 8, 3, padding=1),
                                    nn.ReLU(inplace=True),
                                    DSConv(8, 32),
                                    nn.ReLU(inplace=True))

        # 64 -> 32
        self.layer2 = nn.Sequential(DSConv(64, 64),
                                    nn.ReLU(inplace=True),
                                    DSConv(64, 64),
                                    nn.ReLU(inplace=True),
                                    DSConv(64, 64),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True))

        self.layer2_extra = nn.Sequential(RConv1(64,64,1, False),
                                          RConv1(64,64,1, False),
                                          DSConv(64, 128))

        # 32 -> 16
        self.layer3 = nn.Sequential(DSConv(64, 128),
                                    nn.ReLU(inplace=True),
                                    DSConv(128, 128),
                                    nn.ReLU(inplace=True),
                                    DSConv(128, 128),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True))

        self.layer3_extra = nn.Sequential(RConv1(128,128,1, False),
                                          RConv1(128,128,1, False),
                                          DSConv(128, 128))

        # 16 -> 8
        self.layer4 = nn.Sequential(DSConv(128, 256),
                                    nn.ReLU(inplace=True),
                                    DSConv(256, 256),
                                    nn.ReLU(inplace=True),
                                    DSConv(256, 256),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True),
                                    DSConv(256, 256),
                                    nn.ReLU(inplace=True),
                                    DSConv(256, 256)
                                    )

    def forward(self, x):
        # x : 256, 384
        x1 = self.layer1_fine(x)
        x1_low = self.layer1_coarse(x)
        x1 = torch.cat((x1, x1_low), dim=1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3) # 256, 8, 12

        e2 = self.layer2_extra(x2) # 128, 32, 48
        e3 = self.layer3_extra(x3) # 128, 16, 24

        return x4, e2, e3
