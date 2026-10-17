"""Configurable training augmentation; torchvision is imported only when used."""
import math

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def validate_augmentation(settings):
    def mapping(value, keys, name):
        if not isinstance(value, dict) or value.keys() != keys:
            raise ValueError(f'{name} requires exactly {sorted(keys)}')
    def number(value, name, low, high, positive=False):
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not low <= value <= high or (positive and value <= low)):
            raise ValueError(f'{name} is outside its valid numeric range')
    def integer(value, name, low):
        if type(value) is not int or value < low:
            raise ValueError(f'{name} must be an integer >= {low}')
    def pair(value, name, high=float('inf')):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f'{name} requires [minimum, maximum]')
        for item in value:
            number(item, name, 0, high, positive=True)
        if value[0] > value[1]:
            raise ValueError(f'{name} minimum cannot exceed maximum')

    mapping(settings, {'image_net_1K', 'cifar10', 'batch'}, 'AUGMENTATION')
    common = {'horizontal_flip_p', 'randaugment_enabled', 'randaugment_num_ops',
              'randaugment_magnitude', 'randaugment_num_magnitude_bins',
              'random_erasing_p', 'random_erasing_scale', 'random_erasing_ratio', 'random_erasing_value'}
    for dataset in ('image_net_1K', 'cifar10'):
        extra = {'crop_scale', 'crop_ratio'} if dataset == 'image_net_1K' else {'random_crop_enabled', 'crop_padding'}
        aug = settings[dataset]
        mapping(aug, common | extra, f'AUGMENTATION.{dataset}')
        for key in ('horizontal_flip_p', 'random_erasing_p'):
            number(aug[key], key, 0, 1)
        if type(aug['randaugment_enabled']) is not bool:
            raise ValueError('randaugment_enabled must be boolean')
        integer(aug['randaugment_num_ops'], 'randaugment_num_ops', 0)
        integer(aug['randaugment_num_magnitude_bins'], 'randaugment_num_magnitude_bins', 2)
        integer(aug['randaugment_magnitude'], 'randaugment_magnitude', 0)
        if aug['randaugment_magnitude'] >= aug['randaugment_num_magnitude_bins']:
            raise ValueError('randaugment_magnitude must be < randaugment_num_magnitude_bins')
        pair(aug['random_erasing_scale'], 'random_erasing_scale', 1)
        pair(aug['random_erasing_ratio'], 'random_erasing_ratio')
        value = aug['random_erasing_value']
        if value != 'random':
            number(value, 'random_erasing_value', -float('inf'), float('inf'))
        if dataset == 'image_net_1K':
            pair(aug['crop_scale'], 'crop_scale', 1)
            pair(aug['crop_ratio'], 'crop_ratio')
        else:
            if type(aug['random_crop_enabled']) is not bool:
                raise ValueError('random_crop_enabled must be boolean')
            integer(aug['crop_padding'], 'crop_padding', 0)
    batch = settings['batch']
    mapping(batch, {'datasets', 'p', 'mixup_alpha', 'cutmix_alpha', 'mixup_weight'}, 'AUGMENTATION.batch')
    if (not isinstance(batch['datasets'], list)
            or any(name not in ('image_net_1K', 'cifar10') for name in batch['datasets'])):
        raise ValueError('batch.datasets must be a list of supported dataset names')
    for key in ('p', 'mixup_weight'):
        number(batch[key], key, 0, 1)
    for key in ('mixup_alpha', 'cutmix_alpha'):
        number(batch[key], key, 0, float('inf'), positive=True)


def build_train_transform(conf):
    from torchvision import transforms as T
    size = conf['input_size']
    settings = conf['AUGMENTATION'][conf['dataset']]
    ops = []
    if conf['dataset'] == 'image_net_1K':
        ops.append(T.RandomResizedCrop(size, scale=tuple(settings['crop_scale']),
                                       ratio=tuple(settings['crop_ratio'])))
    elif settings['random_crop_enabled']:
        ops.append(T.RandomCrop(32, padding=settings['crop_padding']))
    ops.append(T.RandomHorizontalFlip(p=settings['horizontal_flip_p']))
    if settings['randaugment_enabled']:
        ops.append(T.RandAugment(num_ops=settings['randaugment_num_ops'],
                                magnitude=settings['randaugment_magnitude'],
                                num_magnitude_bins=settings['randaugment_num_magnitude_bins']))
    if conf['dataset'] == 'cifar10':
        ops.append(T.Resize((size, size)))
    ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
    if settings['random_erasing_p'] > 0:
        ops.append(T.RandomErasing(p=settings['random_erasing_p'],
                                  scale=tuple(settings['random_erasing_scale']),
                                  ratio=tuple(settings['random_erasing_ratio']),
                                  value=settings['random_erasing_value']))
    return T.Compose(ops)


def build_batch_mix(conf):
    settings = conf['AUGMENTATION']['batch']
    if conf['dummy'] or conf['dataset'] not in settings['datasets'] or settings['p'] == 0:
        return None
    from torchvision.transforms import v2
    classes = 10 if conf['dataset'] == 'cifar10' else 1000
    weight = settings['mixup_weight']
    return v2.RandomChoice([
        v2.MixUp(alpha=settings['mixup_alpha'], num_classes=classes),
        v2.CutMix(alpha=settings['cutmix_alpha'], num_classes=classes),
    ], p=[weight, 1 - weight])
