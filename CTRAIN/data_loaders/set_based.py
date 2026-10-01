"""Dataset loading for reproduction presets; downloads are explicitly opt-in."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms as T
from torchvision.datasets.folder import default_loader


def load_set_based_data(dataset, data_root='data', batch_size=128, workers=0,
                        download=False, selection_split='none', validation_fraction=.1,
                        seed=42, smoke=False, input_size=None, normalise=False):
    if dataset not in ('mnist', 'cifar10', 'svhn', 'tinyimagenet'):
        raise ValueError('Unknown reproduction dataset')
    if selection_split not in ('none', 'test', 'validation') or not 0 < validation_fraction < 1:
        raise ValueError('Invalid selection split or holdout fraction')
    tiny = dataset == 'tinyimagenet'
    size, channels, classes = (28, 1, 10) if dataset == 'mnist' else (64, 3, 200) if tiny else (32, 3, 10)
    size = input_size or size
    # CTRAIN defaults: no loader normalization; radii and pixels are both in [0, 1].
    mean = [.4802, .4481, .3975] if tiny else [.4914, .4822, .4465]
    std = [.2302, .2265, .2262] if tiny else [.2023, .1994, .2010]
    tensor = [T.ToTensor(), T.Normalize(mean, std)] if normalise else [T.ToTensor()]
    test_transform = T.Compose([T.CenterCrop(size), *tensor])
    padding = (4 if tiny and size == 64 else 0 if tiny else 2) if normalise else 4
    augmentation = [T.RandomCrop(size, padding, padding_mode='edge'), T.RandomHorizontalFlip()]
    if normalise:
        augmentation.reverse()
    train_transform = T.Compose([*augmentation, *tensor]) if dataset in ('cifar10', 'tinyimagenet') else test_transform
    root = Path(data_root)
    if smoke:
        train = datasets.FakeData(2, (channels, size, size), classes, train_transform)
        plain_train = datasets.FakeData(2, (channels, size, size), classes, test_transform)
        test = plain_train
    elif tiny:
        if (root / 'tiny-imagenet-200').exists():
            root = root / 'tiny-imagenet-200'
        train = datasets.ImageFolder(root / 'train', train_transform)
        plain_train = datasets.ImageFolder(root / 'train', test_transform)
        annotations, images = root / 'val' / 'val_annotations.txt', root / 'val' / 'images'
        if annotations.exists() and any(images.glob('*.JPEG')):
            class OfficialValidation(Dataset):
                def __init__(self):
                    self.samples = [(images / row.split()[0], train.class_to_idx[row.split()[1]])
                                    for row in annotations.read_text().splitlines()]

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, index):
                    path, label = self.samples[index]
                    return test_transform(default_loader(path)), label
            test = OfficialValidation()
        else:
            test = datasets.ImageFolder(images if images.exists() else root / 'val', test_transform)
            if train.class_to_idx != test.class_to_idx:
                raise ValueError('TinyImageNet class mappings differ')
    elif dataset == 'svhn':
        train = datasets.SVHN(root, split='train', download=download, transform=train_transform)
        plain_train = datasets.SVHN(root, split='train', download=False, transform=test_transform)
        test = datasets.SVHN(root, split='test', download=download, transform=test_transform)
    else:
        constructor = datasets.MNIST if dataset == 'mnist' else datasets.CIFAR10
        train = constructor(root, train=True, download=download, transform=train_transform)
        plain_train = constructor(root, train=True, download=False, transform=test_transform)
        test = constructor(root, train=False, download=download, transform=test_transform)
    selection = test if selection_split == 'test' else None
    if selection_split == 'validation':
        indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
        count = max(1, int(len(train) * validation_fraction))
        if count >= len(train):
            raise ValueError('Holdout leaves no training samples')
        selection = Subset(plain_train, indices[:count].tolist())
        train = Subset(train, indices[count:].tolist())

    def loader(data, shuffle=False):
        result = DataLoader(data, batch_size=2 if smoke else batch_size, shuffle=shuffle,
                            num_workers=workers)
        result.normalised = normalise
        result.mean, result.std = (torch.tensor(mean), torch.tensor(std)) if normalise else (torch.zeros(channels), torch.ones(channels))
        result.min = (-result.mean / result.std).reshape(channels, 1, 1)
        result.max = ((1 - result.mean) / result.std).reshape(channels, 1, 1)
        return result
    return loader(train, True), None if selection is None else loader(selection), loader(test)
