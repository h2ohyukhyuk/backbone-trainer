"""timm MNASNet with APAG0-style post-downsampling feature branches."""
import torch.nn as nn
import timm
from timm.models import FeatureInfo

from model.layers import DSConv, RConv1


class APAGMnas(nn.Module):
    def __init__(self, backbone_name='mnasnet_050', pretrained=False):
        super().__init__()
        aliases = {
            'mnasnet0_5': 'mnasnet_050',
            'mnasnet0.5': 'mnasnet_050',
            'mnasnet0_75': 'mnasnet_075',
            'mnasnet0.75': 'mnasnet_075',
            'mnasnet1_0': 'mnasnet_100',
            'mnasnet1.0': 'mnasnet_100',
        }
        backbone_name = aliases.get(backbone_name, backbone_name)
        if backbone_name not in ('mnasnet_050', 'mnasnet_075', 'mnasnet_100'):
            raise ValueError('backbone_name must be mnasnet_050, mnasnet_075 or mnasnet_100')
        self.backbone_name = backbone_name
        backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True, out_indices=(2, 3, 4),
        )
        # Stock timm exposes only the final /32 stage (blocks.6). Also expose
        # the existing preceding /32 stage (blocks.5); add no new computation.
        feature_info = list(backbone.feature_info.info)
        feature_info.insert(-1, dict(
            num_chs=backbone.blocks[5][-1].conv_pwl.out_channels,
            reduction=32,
            module='blocks.5',
            stage=6,
        ))
        backbone.feature_info = FeatureInfo(feature_info, out_indices=(2, 3, 4, 5))
        # EfficientNetFeatures selects existing stage outputs with this lookup.
        # Keep its module hierarchy so backbone checkpoint keys do not change.
        backbone._stage_out_idx = {
            feature['stage']: index
            for index, feature in enumerate(backbone.feature_info.get_dicts())
        }
        self.backbone = backbone
        c1, c2, c3, c4 = self.backbone.feature_info.channels()
        self.out_channels = (c4, 128, 128, 128)

        # Adapt channels before DSConv: its grouped skip requires divisibility.
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
        # Branch from /32 after downsampling, just as /8 and /16 above.
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
