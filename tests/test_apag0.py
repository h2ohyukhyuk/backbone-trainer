"""Guard APAGNet0 checkpoint layout after extracting the inherited backbone."""
import hashlib
import json
import unittest

import torch

from model.APAG0 import APAG0
from net.APAGNet0 import Network


class APAG0CompatibilityTests(unittest.TestCase):
    def test_pre_refactor_checkpoint_schema_and_optimizer_parameter_order(self):
        model = Network()
        # Snapshots from the original, monolithic net/APAGNet0.py (1000 classes).
        # Include BN buffers as well as parameters; a new backbone prefix must fail.
        schema = [(name, list(value.shape), str(value.dtype))
                  for name, value in model.state_dict().items()]
        parameters = [(name, list(value.shape), str(value.dtype))
                      for name, value in model.named_parameters()]
        self.assertEqual(len(schema), 294)
        self.assertEqual(len(parameters), 186)
        self.assertEqual(
            hashlib.sha256(json.dumps(schema).encode()).hexdigest(),
            '8b751991c1b82a31f0c4c324723acf6c60a4d28131124a4cf225feed13697cf3',
        )
        self.assertEqual(
            hashlib.sha256(json.dumps(parameters).encode()).hexdigest(),
            '588ea7c179aa6b0ed9c34b22afc39f36e74767281938c6170583aa2ba96cc7bb',
        )

    def test_standalone_backbone_matches_inherited_features(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            network = Network(num_classes=7).eval()
            backbone = APAG0().eval()
            # A classifier-free checkpoint needs no key renaming to use the backbone.
            backbone_state = {name: value for name, value in network.state_dict().items()
                              if not name.startswith('classifier.')}
            backbone.load_state_dict(backbone_state, strict=True)
            self.assertIsInstance(network, APAG0)
            self.assertIs(Network.forward_features, APAG0.forward_features)
            images = torch.randn(1, 3, 256, 384)
            with torch.no_grad():
                expected = network.forward_features(images)
                actual = backbone(images)
                for left, right in zip(expected, actual):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                self.assertEqual(network(images).shape, (1, 7))
            self.assertEqual([tuple(value.shape) for value in actual], [
                (1, 256, 8, 12), (1, 128, 32, 48),
                (1, 128, 16, 24), (1, 128, 8, 12),
            ])
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
