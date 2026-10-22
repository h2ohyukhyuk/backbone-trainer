"""Verify ResNet feature extraction and all APAG branches without training."""
from pathlib import Path
import unittest

import timm
import torch
import yaml

from model.APAGRes import APAGRes
from net.APAGResNet import Network


VARIANTS = ('resnet18', 'resnet50', 'resnet101')


class APAGResTests(unittest.TestCase):
    def setUp(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)

    def test_extraction_matches_stock_features_and_first_layer4_block(self):
        for name in VARIANTS:
            with self.subTest(backbone=name):
                actual = APAGRes(name).backbone.eval()
                stock = timm.create_model(name, pretrained=False, features_only=True,
                                          out_indices=(2, 3, 4)).eval()
                stock.load_state_dict(actual.state_dict(), strict=True)
                intermediate = []
                handle = stock.layer4[0].register_forward_hook(
                    lambda module, args, output: intermediate.append(output))
                try:
                    images = torch.randn(1, 3, 64, 96)
                    with torch.no_grad():
                        outputs = actual(images)
                        reference = stock(images)
                finally:
                    handle.remove()
                self.assertEqual(actual.feature_info.reduction(), [8, 16, 32, 32])
                self.assertFalse(hasattr(actual, 'fc'))
                self.assertFalse(hasattr(actual, 'global_pool'))
                self.assertEqual(len(outputs), 4)
                for index, expected in zip((0, 1, 3), reference):
                    torch.testing.assert_close(outputs[index], expected, rtol=0, atol=0)
                torch.testing.assert_close(outputs[2], intermediate[0], rtol=0, atol=0)
                self.assertIsNot(outputs[2], outputs[3])

    def test_extras_receive_post_downsampling_features(self):
        for name in VARIANTS:
            with self.subTest(backbone=name):
                model = APAGRes(name).eval()
                recorded = {}

                def record_input(key):
                    def hook(module, args):
                        recorded[key] = args[0]
                    return hook

                def record_outputs(module, args, outputs):
                    recorded['backbone'] = outputs

                handles = [model.backbone.register_forward_hook(record_outputs)]
                keys = ('layer2_extra', 'layer3_extra', 'layer4_extra')
                handles += [getattr(model, key).register_forward_pre_hook(record_input(key))
                            for key in keys]
                try:
                    with torch.no_grad():
                        features = model(torch.randn(1, 3, 64, 96))
                finally:
                    for handle in handles:
                        handle.remove()
                channels = 512 if name == 'resnet18' else 2048
                self.assertEqual(model.out_channels, (channels, 128, 128, 128))
                self.assertEqual([tuple(t.shape) for t in features], [
                    (1, channels, 2, 3), (1, 128, 8, 12),
                    (1, 128, 4, 6), (1, 128, 2, 3),
                ])
                self.assertIs(features[0], recorded['backbone'][3])
                for index, key in enumerate(keys):
                    self.assertIs(recorded[key], recorded['backbone'][index])
                self.assertFalse(hasattr(model, 'layer5'))

    def test_experiment_configs_and_classifier_gradients(self):
        root = Path(__file__).resolve().parents[1]
        for name in VARIANTS:
            with self.subTest(backbone=name):
                path = root / 'experiments' / f'APAGResNet_{name}_imagenet.yml'
                config = yaml.safe_load(path.read_text())
                self.assertEqual(config['arch'], 'APAGResNet')
                self.assertEqual(config['model_kwargs']['backbone_name'], name)
                model = Network(num_classes=7, **config['model_kwargs']).train()
                self.assertEqual(model.classifier.num_inputs, 4)
                logits = model(torch.randn(2, 3, 64, 96))
                self.assertEqual(logits.shape, (2, 7))
                logits.square().mean().backward()
                for key in ('backbone', 'layer2_extra', 'layer3_extra', 'layer4_extra'):
                    gradients = [p.grad for p in getattr(model.apagres, key).parameters()]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all()
                                        for g in gradients))
                    self.assertTrue(any(torch.count_nonzero(g) > 0 for g in gradients))
                for head in model.classifier.heads:
                    self.assertIsNotNone(head[-1].weight.grad)

    def test_invalid_variant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'resnet18, resnet50 or resnet101'):
            APAGRes('resnet52')


if __name__ == '__main__':
    unittest.main()
