"""Original timm ResNet backbone with a separate average pool and FC head.

After loading a training checkpoint, reuse ``model.backbone`` directly or call
``model.forward_features(images)`` to obtain an unpooled NCHW feature map.
``model.backbone.state_dict()`` contains only the feature extractor weights.
"""
import torch
import torch.nn as nn
import timm


class Network(nn.Module):
    def __init__(self, num_classes=1000, backbone_name='resnet18',
                 backbone_pretrained=False):
        super().__init__()
        if backbone_name not in ('resnet18', 'resnet50', 'resnet101'):
            raise ValueError('backbone_name must be resnet18, resnet50, resnet101')
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name, pretrained=backbone_pretrained)
        self.out_channels = self.backbone.num_features
        # Load optional pretrained feature weights before replacing the head.
        # Empty pooling plus an identity classifier also lets backbone(x)
        # return the unpooled feature map for transfer learning.
        self.backbone.reset_classifier(0, global_pool='')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.out_channels, num_classes)

    def forward_features(self, x):
        """Return the original backbone's final spatial feature map."""
        return self.backbone(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


if __name__ == '__main__':
    model = Network().eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 256, 384))
    print('output.shape:', output.shape)
