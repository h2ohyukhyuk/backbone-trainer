"""Check fused feature shapes and end-to-end classifier gradients."""
import unittest

import torch

from model.APAGRes import APAGRes
from model.DAPRes import DAPRes
from net.DAPResNet import Network


class DAPResTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_apagres_weights_and_fused_features(self):
        for name in ('resnet18', 'resnet50', 'resnet101'):
            with self.subTest(backbone=name):
                reference = APAGRes(name).eval()
                model = DAPRes(name).eval()
                result = model.load_state_dict(reference.state_dict(), strict=False)
                self.assertFalse(result.unexpected_keys)
                self.assertTrue(all(k.startswith('fpn.') for k in result.missing_keys))
                images = torch.randn(1, 3, 64, 96)
                with torch.no_grad():
                    x4, e2, e3, e4 = reference(images)
                    expected = model.fpn(e2, e3, e4, x4)
                    actual = model(images)
                self.assertEqual(actual.shape, (1, 128, 8, 12))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_classifier_and_gradients(self):
        for name in ('resnet18', 'resnet50', 'resnet101'):
            with self.subTest(backbone=name):
                model = Network(num_classes=7, backbone_name=name).train()
                logits = model(torch.randn(2, 3, 64, 96))
                self.assertEqual(logits.shape, (2, 7))
                logits.square().mean().backward()
                for key in ('backbone', 'layer2_extra', 'layer3_extra',
                            'layer4_extra', 'fpn'):
                    grads = [p.grad for p in getattr(model.dapres, key).parameters()]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all()
                                        for g in grads))
                    self.assertTrue(any(torch.count_nonzero(g) > 0 for g in grads))
                self.assertIsNotNone(model.fc.weight.grad)

    def test_invalid_backbone(self):
        with self.assertRaises(ValueError):
            DAPRes('resnet52')


if __name__ == '__main__':
    unittest.main()
