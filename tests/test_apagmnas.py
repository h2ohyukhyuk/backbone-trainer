"""Check APAG0-style post-downsampling MNASNet branches without training."""
import unittest

import torch
import timm
from torch import nn

from model.APAGMnas import APAGMnas
from net.APAGMNasNet import Network


class APAGMnasTests(unittest.TestCase):
    def setUp(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)

    def test_075_aliases(self):
        for name in ('mnasnet0_75', 'mnasnet0.75'):
            with self.subTest(backbone=name):
                model = APAGMnas(name, pretrained=False)
                self.assertEqual(model.backbone_name, 'mnasnet_075')
                self.assertEqual(model.out_channels, (240, 128, 128, 128))

    def test_post_stride_extras_and_unmodified_backbone_output(self):
        for name in ('mnasnet_050', 'mnasnet_075', 'mnasnet_100'):
            with self.subTest(backbone=name):
                backbone = APAGMnas(name, pretrained=False).eval()
                inputs = {}
                def record_input(key):
                    def hook(module, args):
                        inputs[key] = args[0]
                    return hook
                def record_backbone(module, args, outputs):
                    inputs['backbone_outputs'] = outputs
                handles = [backbone.backbone.register_forward_hook(record_backbone)]
                handles += [getattr(backbone, key).register_forward_pre_hook(record_input(key))
                           for key in ('layer2_extra', 'layer3_extra', 'layer4_extra')]
                try:
                    with torch.no_grad():
                        features = backbone(torch.randn(1, 3, 256, 384))
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(backbone.backbone.feature_info.reduction(), [8, 16, 32, 32])
                channels = {'mnasnet_050': 160, 'mnasnet_075': 240, 'mnasnet_100': 320}[name]
                self.assertEqual(backbone.out_channels, (channels, 128, 128, 128))
                self.assertFalse(hasattr(backbone, 'layer5'))
                self.assertEqual([tuple(value.shape) for value in features], [
                    (1, channels, 8, 12), (1, 128, 32, 48),
                    (1, 128, 16, 24), (1, 128, 8, 12),
                ])
                self.assertEqual(inputs['layer2_extra'].shape[-2:], (32, 48))
                self.assertEqual(inputs['layer3_extra'].shape[-2:], (16, 24))
                self.assertEqual(inputs['layer4_extra'].shape[-2:], (8, 12))
                self.assertEqual(len(inputs['backbone_outputs']), 4)
                self.assertIs(features[0], inputs['backbone_outputs'][3])
                for index, key in enumerate(('layer2_extra', 'layer3_extra', 'layer4_extra')):
                    self.assertIs(inputs[key], inputs['backbone_outputs'][index])
                self.assertIsNot(features[0], inputs['layer4_extra'])
                early_channels = {'mnasnet_050': 96, 'mnasnet_075': 144, 'mnasnet_100': 192}[name]
                self.assertEqual(inputs['layer4_extra'].shape[1], early_channels)
                for key in ('layer2_extra', 'layer3_extra', 'layer4_extra'):
                    self.assertIsInstance(getattr(backbone, key)[-1], nn.ReLU)

    def test_backbone_retains_stock_weights_and_features(self):
        for name in ('mnasnet_050', 'mnasnet_075', 'mnasnet_100'):
            with self.subTest(backbone=name):
                actual = APAGMnas(name, pretrained=False).backbone.eval()
                stock = timm.create_model(name, pretrained=False, features_only=True,
                                          out_indices=(2, 3, 4)).eval()
                # Extraction adds no parameters and keeps timm's checkpoint key layout.
                stock.load_state_dict(actual.state_dict(), strict=True)
                images = torch.randn(1, 3, 64, 96)
                with torch.no_grad():
                    four_outputs = actual(images)
                    three_outputs = stock(images)
                for actual_index, reference in zip((0, 1, 3), three_outputs):
                    torch.testing.assert_close(four_outputs[actual_index], reference, rtol=0, atol=0)

    def test_classifier_uses_all_four_branches(self):
        for name in ('mnasnet_050', 'mnasnet_075', 'mnasnet_100'):
            with self.subTest(backbone=name):
                model = Network(num_classes=7, backbone_name=name).train()
                self.assertEqual(model.classifier.num_inputs, 4)
                logits = model(torch.randn(2, 3, 64, 96))
                self.assertEqual(logits.shape, (2, 7))
                logits.square().mean().backward()
                # Each branch must contribute to the classifier/loss.
                for key in ('layer2_extra', 'layer3_extra', 'layer4_extra', 'backbone'):
                    gradients = [p.grad for p in getattr(model.apagmnas, key).parameters()]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                    self.assertTrue(any(torch.count_nonzero(g) > 0 for g in gradients))
                for head in model.classifier.heads:
                    self.assertIsNotNone(head[-1].weight.grad)


if __name__ == '__main__':
    unittest.main()
