"""Standard image loaders preserve splits, coordinates, and reproduction inputs."""
from functools import partial
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from CTRAIN.data_loaders import load_mnist, load_cifar10, load_svhn, load_tinyimagenet


LOADERS = ((load_mnist, 'MNIST', 1), (load_cifar10, 'CIFAR10', 3),
           (load_svhn, 'SVHN', 3), (load_tinyimagenet, 'ImageFolder', 3))


class Images(Dataset):
    def __init__(self, channels, transform, **kwargs):
        self.channels, self.transform = channels, transform
        self.class_to_idx = {'a': 0, 'b': 1}

    def __len__(self):
        return 20

    def __getitem__(self, index):
        shape = (64, 64) if self.channels == 1 else (64, 64, self.channels)
        image = Image.fromarray(np.zeros(shape, dtype=np.uint8))
        return self.transform(image), index % 2


def white(image):
    return Image.fromarray(np.full_like(np.asarray(image), 255))


def defaults_and_splits():
    for load, constructor, channels in LOADERS:
        with patch(f'CTRAIN.data_loaders.data_loaders.datasets.{constructor}',
                   side_effect=partial(Images, channels)) as dataset, \
             patch('CTRAIN.data_loaders.data_loaders.download_tinyimagenet'):
            # Original positional API and 80/20 split still work.
            train, validation, test = load(4, False, [white], True, '/unused')
            assert (len(train.dataset), len(validation.dataset), len(test.dataset)) == (16, 4, 20)
            assert train.dataset[0][0].mean() == 1
            assert validation.dataset[0][0].mean() == test.dataset[0][0].mean() == 0
            assert train.num_workers == (4 if constructor == 'ImageFolder' else 2)
            assert validation.num_workers == 2
            assert train.min.shape == (1, channels, 1, 1)
            assert len(load(4, False, [], False, '/unused')) == 2
            dataset.reset_mock()
            options = dict(workers=0, download=False, normalise=False,
                           train_transforms=[white], selection_split='validation',
                           validation_fraction=.25, seed=13)
            first = load(**options)
            second = load(**options)
            train, validation, test = first
            assert len(train.dataset) == 15 and len(validation.dataset) == 5
            expected = torch.randperm(20, generator=torch.Generator().manual_seed(13)).tolist()
            assert validation.dataset.indices == expected[:5]
            assert train.dataset.indices == expected[5:] == second[0].dataset.indices
            assert not set(train.dataset.indices) & set(validation.dataset.indices)
            assert validation.dataset.dataset.transform is test.dataset.transform
            assert validation.dataset[0][0].mean() == 0
            assert all(loader.num_workers == 0 for loader in first)
            if constructor != 'ImageFolder':
                assert all(call.kwargs['download'] is False for call in dataset.call_args_list)
            _, selection, test = load(workers=0, selection_split='test', download=False)
            assert selection is test
            _, selection, _ = load(workers=0, selection_split='none', download=False)
            assert selection is None


def coordinates_and_validation():
    for load, _, channels in LOADERS:
        for normalise in (False, True):
            train, selection, test = load(smoke=True, workers=0, download=False,
                                          normalise=normalise, selection_split='none')
            images, _ = next(iter(test))
            assert len(train.dataset) == len(test.dataset) == 2 and selection is None
            assert torch.all(images >= test.min) and torch.all(images <= test.max)
            assert test.std.shape == (channels,)
    _, test = load_mnist(smoke=True, workers=0, val_split=False)
    torch.testing.assert_close(test.mean, torch.tensor([.1307]))
    torch.testing.assert_close(test.std, torch.tensor([.3081]))
    stats = dict(mean=[.25], std=[.5])
    _, test = load_mnist(smoke=True, workers=0, val_split=False, normalisation_stats=stats)
    torch.testing.assert_close(test.min, torch.tensor([[[[-.5]]]]))
    torch.testing.assert_close(test.max, torch.tensor([[[[1.5]]]]))
    invalid = [dict(selection_split='bad'), dict(validation_fraction=0),
               dict(validation_fraction=1), dict(input_size=0), dict(input_size=True),
               dict(normalisation_stats=dict(mean=[0.], std=[0.])),
               dict(normalisation_stats=dict(mean=[float('nan')], std=[1.])),
               dict(normalisation_stats=dict(mean=[0., 0.], std=[1., 1.])),
               dict(normalise=False, normalisation_stats=stats)]
    for options in invalid:
        with patch('CTRAIN.data_loaders.data_loaders.datasets.MNIST',
                   side_effect=AssertionError('Invalid options must fail before loading')):
            try:
                load_mnist(**options)
            except ValueError:
                pass
            else:
                raise AssertionError(options)
    with patch.object(Images, '__len__', return_value=1), \
         patch('CTRAIN.data_loaders.data_loaders.datasets.MNIST',
               side_effect=partial(Images, 1)):
        try:
            load_mnist(selection_split='validation', workers=0, download=False)
        except ValueError as error:
            assert 'no training samples' in str(error)
        else:
            raise AssertionError('Empty training split accepted')


def tinyimagenet_layouts():
    for layout in ('official', 'folders', 'images_folders'):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'tiny-imagenet-200'
            image = Image.new('RGB', (64, 64), color=(64, 128, 192))
            for name in ('a', 'b'):
                folder = root / 'train' / name
                folder.mkdir(parents=True)
                image.save(folder / 'train.JPEG')
            val = root / 'val'
            if layout == 'official':
                (val / 'images').mkdir(parents=True)
                for name in ('a', 'b'):
                    image.save(val / 'images' / f'{name}.JPEG')
                (val / 'val_annotations.txt').write_text('b.JPEG\tb\t0\t0\t64\t64\na.JPEG\ta\t0\t0\t64\t64\n')
            else:
                for name in ('a', 'b'):
                    folder = (val / 'images' if layout == 'images_folders' else val) / name
                    folder.mkdir(parents=True)
                    image.save(folder / 'val.JPEG')
            before = sorted(path.relative_to(root) for path in root.rglob('*'))
            for data_root in (directory, root):
                with patch('CTRAIN.data_loaders.data_loaders.download_tinyimagenet',
                           side_effect=AssertionError('Must not download existing data')):
                    train, test = load_tinyimagenet(data_root=data_root, download=False,
                                                    val_split=False, workers=0,
                                                    input_size=56, train_transforms=[])
                images, labels = next(iter(test))
                assert images.shape == (2, 3, 56, 56)
                assert labels.tolist() == ([1, 0] if layout == 'official' else [0, 1])
            assert before == sorted(path.relative_to(root) for path in root.rglob('*'))
            if layout == 'folders':
                (val / 'b').rename(val / 'c')
                try:
                    load_tinyimagenet(data_root=root, download=False, workers=0)
                except ValueError as error:
                    assert 'class mappings differ' in str(error)
                else:
                    raise AssertionError('Mismatched validation classes accepted')


def main():
    torch.set_num_threads(1)
    defaults_and_splits()
    coordinates_and_validation()
    tinyimagenet_layouts()
    print('Standard data loader checks passed.')


if __name__ == '__main__':
    main()
