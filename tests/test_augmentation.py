import copy
import importlib.util
import unittest
from unittest.mock import patch

import torch
import main
from dataset.augmentation import build_train_transform, build_batch_mix


class AugmentationValidationTests(unittest.TestCase):
    def test_invalid_parameters_are_rejected(self):
        baseline = main.read_config('experiments/config_default.yaml')
        for section, key, value in [
            ('batch', 'p', 1.1), ('batch', 'mixup_alpha', 0),
            ('batch', 'mixup_weight', -1), ('batch', 'datasets', ['unknown']),
            ('image_net_1K', 'crop_scale', [.8, .1]),
            ('image_net_1K', 'crop_ratio', [0, 1]),
            ('image_net_1K', 'randaugment_magnitude', 31),
            ('cifar10', 'random_crop_enabled', 1),
            ('cifar10', 'random_erasing_p', float('nan')),
        ]:
            with self.subTest(section=section, key=key):
                conf = copy.deepcopy(baseline)
                conf['AUGMENTATION'][section][key] = value
                with self.assertRaises(ValueError):
                    main.validate_config(conf)


@unittest.skipUnless(importlib.util.find_spec('torchvision'), 'requires torchvision')
class AugmentationBehaviorTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from PIL import Image
        self.conf = vars(main.parse_args([]))
        self.conf.update(dataset='cifar10', dummy=False, input_size=32,
                         distributed=False, no_accel=True, download=False)
        self.image = Image.fromarray(np.arange(32*32*3, dtype=np.uint8).reshape(32, 32, 3))

    def test_flip_probability_controls_pixels(self):
        conf = self.conf
        aug = conf['AUGMENTATION']['cifar10']
        aug.update(random_crop_enabled=False, horizontal_flip_p=0)
        original = build_train_transform(conf)(self.image)
        aug['horizontal_flip_p'] = 1
        flipped = build_train_transform(conf)(self.image)
        torch.testing.assert_close(flipped, original.flip(-1))

    def test_erasing_probability_and_value_are_applied(self):
        aug = self.conf['AUGMENTATION']['cifar10']
        aug.update(random_crop_enabled=False, horizontal_flip_p=0,
                   random_erasing_p=1, random_erasing_scale=[.2, .2], random_erasing_ratio=[1., 1.],
                   random_erasing_value=7.)
        result = build_train_transform(self.conf)(self.image)
        self.assertTrue((result == 7.).all(dim=0).any())

    def test_imagenet_and_cifar_pipelines_accept_configured_randaugment(self):
        for dataset in ('image_net_1K', 'cifar10'):
            with self.subTest(dataset=dataset):
                conf = copy.deepcopy(self.conf)
                conf['dataset'] = dataset
                conf['AUGMENTATION'][dataset].update(randaugment_enabled=True,
                                                     randaugment_num_ops=1, randaugment_magnitude=3)
                image = build_train_transform(conf)(self.image)
                self.assertEqual(tuple(image.shape), (3, 32, 32))
                self.assertTrue(torch.isfinite(image).all())

    def test_batch_mix_supports_both_choices_and_dataset_gating(self):
        self.assertIsNone(build_batch_mix(self.conf))  # CIFAR-10 disabled by default.
        batch = self.conf['AUGMENTATION']['batch']
        batch.update(datasets=['cifar10'], p=1.)
        for weight in (0., 1.):
            batch['mixup_weight'] = weight
            mixed_images, labels = build_batch_mix(self.conf)(torch.randn(4, 3, 32, 32), torch.arange(4))
            self.assertEqual(tuple(mixed_images.shape), (4, 3, 32, 32))
            self.assertEqual(tuple(labels.shape), (4, 10))
            torch.testing.assert_close(labels.sum(-1), torch.ones(4))
        batch['p'] = 0
        self.assertIsNone(build_batch_mix(self.conf))
        batch['p'] = 1
        self.conf['dummy'] = True
        self.assertIsNone(build_batch_mix(self.conf))

    def test_validation_is_deterministic_despite_training_augmentation(self):
        from dataset.cifar10 import _build_dataset
        with patch('torchvision.datasets.CIFAR10') as dataset:
            _build_dataset(self.conf, train=False)
        transform = dataset.call_args.kwargs['transform']
        torch.testing.assert_close(transform(self.image), transform(self.image))
        # Training settings do not inject random transforms into validation.
        self.assertFalse(any(type(op).__name__.startswith('Random') for op in transform.transforms))

    def test_trainer_uses_configured_batch_probability(self):
        from torch.utils.data import DataLoader, TensorDataset
        from train_val.classification import Trainer
        conf = self.conf
        conf.update(start_epoch=0, print_freq=100)
        conf['AUGMENTATION']['batch'].update(datasets=['cifar10'], p=1., mixup_weight=1.)
        loader = DataLoader(TensorDataset(torch.randn(4, 3, 32, 32), torch.arange(4)), batch_size=4)
        model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 10))
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        trainer = Trainer(conf, loader, optimizer, None)
        with patch.object(trainer, 'batch_mix', wraps=trainer.batch_mix) as mixing:
            loss = trainer(model, 1)
            self.assertEqual(mixing.call_count, 1)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))


if __name__ == '__main__':
    unittest.main()
