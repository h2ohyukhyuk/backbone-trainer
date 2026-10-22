"""Check original ResNet/MNASNet behavior and independent backbone reuse."""
import importlib
from pathlib import Path
import unittest

import timm
import torch
import yaml


VARIANTS = (
    ('ResNet', 'resnet18', 512),
    ('ResNet', 'resnet50', 2048),
    ('ResNet', 'resnet101', 2048),
    ('MNasNet', 'mnasnet_050', 1280),
    ('MNasNet', 'mnasnet_075', 1280),
    ('MNasNet', 'mnasnet_100', 1280),
)


class PlainBackboneTests(unittest.TestCase):
    def setUp(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)

    def test_matches_original_classifier_and_reusable_feature_weights(self):
        for family, name, channels in VARIANTS:
            with self.subTest(backbone=name):
                cls = importlib.import_module(f'net.{family}').Network
                model = cls(num_classes=7, backbone_name=name).eval()
                stock = timm.create_model(name, pretrained=False, num_classes=7).eval()
                # Load the original network's backbone and classifier separately.
                classifier_key = 'fc' if family == 'ResNet' else 'classifier'
                state = {k: v for k, v in stock.state_dict().items()
                         if not k.startswith(classifier_key + '.')}
                model.backbone.load_state_dict(state, strict=True)
                model.fc.load_state_dict(stock.get_classifier().state_dict(), strict=True)
                self.assertEqual(model.out_channels, channels)
                self.assertIsInstance(model.backbone.get_classifier(), torch.nn.Identity)
                self.assertFalse(any(k.startswith(('fc.', 'classifier.'))
                                     for k in model.backbone.state_dict()))
                images = torch.randn(1, 3, 64, 96)
                with torch.no_grad():
                    actual_features = model.forward_features(images)
                    reference_features = stock.forward_features(images)
                    torch.testing.assert_close(actual_features, reference_features, rtol=0, atol=0)
                    self.assertEqual(actual_features.shape, (1, channels, 2, 3))
                    torch.testing.assert_close(model(images), stock(images), rtol=0, atol=0)
                # Backbone-only weights are directly reusable by an independent
                # stock model after removing its pooling/classifier.
                stock.reset_classifier(0, global_pool='')
                stock.load_state_dict(model.backbone.state_dict(), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(stock(images), actual_features, rtol=0, atol=0)
                if family == 'MNasNet':
                    # Preserve the original convolutional head before pooling.
                    self.assertIn('conv_head.weight', model.backbone.state_dict())

    def test_configs_support_training_and_backbone_freezing(self):
        root = Path(__file__).resolve().parents[1]
        for family, name, channels in VARIANTS:
            with self.subTest(backbone=name):
                path = root / 'experiments' / f'{family}_{name}_imagenet.yml'
                config = yaml.safe_load(path.read_text())
                self.assertEqual(config['arch'], family)
                self.assertEqual(config['model_kwargs']['backbone_name'], name)
                cls = importlib.import_module(f'net.{config["arch"]}').Network
                model = cls(num_classes=7, **config['model_kwargs']).train()
                logits = model(torch.randn(2, 3, 64, 96))
                self.assertEqual(logits.shape, (2, 7))
                torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3])).backward()
                gradients = [p.grad for p in model.backbone.parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertTrue(any(torch.count_nonzero(g) > 0 for g in gradients))
                self.assertTrue(torch.isfinite(model.fc.weight.grad).all())
                model.zero_grad(set_to_none=True)
                model.backbone.requires_grad_(False)
                model.backbone.eval()
                model(torch.randn(2, 3, 64, 96)).sum().backward()
                self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
                self.assertIsNotNone(model.fc.weight.grad)

    def test_mnasnet_alias_and_invalid_variants(self):
        from net.MNasNet import Network as MNasNet
        from net.ResNet import Network as ResNet
        model = MNasNet(backbone_name='mnasnet0.75')
        self.assertEqual(model.backbone_name, 'mnasnet_075')
        with self.assertRaisesRegex(ValueError, 'backbone_name'):
            MNasNet(backbone_name='resnet18')
        with self.assertRaisesRegex(ValueError, 'backbone_name'):
            ResNet(backbone_name='resnet52')


if __name__ == '__main__':
    unittest.main()
