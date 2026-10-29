from CTRAIN.data_loaders.preprocess_tinyimagenet import create_val_img_folder
import copy
import os
from pathlib import Path
import zipfile

import numpy as np
import requests
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

MNIST_MEAN, MNIST_STD = .1307, .3081


class _TinyImageNetValidation(Dataset):
    """Read official validation annotations without reorganising images."""

    def __init__(self, root, class_to_idx, transform):
        self.transform = transform
        self.samples = [(root / 'images' / row.split()[0], class_to_idx[row.split()[1]])
                        for row in (root / 'val_annotations.txt').read_text().splitlines()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(default_loader(path)), label


def _load_image_data(dataset, batch_size, normalise, train_transforms, val_split,
                     data_root, workers, download, selection_split,
                     validation_fraction, seed, smoke, input_size, normalisation_stats):
    """Construct shared image transforms, selection splits and metadata-bearing loaders."""
    if selection_split not in (None, 'none', 'test', 'validation'):
        raise ValueError('Invalid selection split')
    if not 0 < validation_fraction < 1:
        raise ValueError('Invalid holdout fraction')
    if input_size is not None and (isinstance(input_size, bool)
                                   or not isinstance(input_size, int) or input_size <= 0):
        raise ValueError('input_size must be a positive integer')
    size, channels, classes = ((28, 1, 10) if dataset == 'mnist' else
                               (64, 3, 200) if dataset == 'tinyimagenet' else (32, 3, 10))
    size = input_size or size
    mean, std = (([MNIST_MEAN], [MNIST_STD]) if dataset == 'mnist' else
                 ([.4802, .4481, .3975], [.2302, .2265, .2262]) if dataset == 'tinyimagenet' else
                 ([.4914, .4822, .4465], [.2023, .1994, .2010]))
    if normalisation_stats is not None:
        mean, std = normalisation_stats['mean'], normalisation_stats['std']
        if (not normalise or len(mean) != channels or len(std) != channels
                or not torch.isfinite(torch.tensor(mean)).all()
                or not torch.isfinite(torch.tensor(std)).all() or min(std) <= 0):
            raise ValueError('normalisation_stats must contain finite channel means and positive deviations with normalise=True')
    mean, std = ((torch.tensor(mean), torch.tensor(std)) if normalise else
                 (torch.zeros(channels), torch.ones(channels)))
    tensor = [transforms.ToTensor()]
    if normalise:
        tensor.append(transforms.Normalize(mean, std))
    crop = [] if input_size is None else [transforms.CenterCrop(size)]
    test_transform = transforms.Compose([*crop, *tensor])
    train_transform = transforms.Compose([*train_transforms, *crop, *tensor])
    root = Path(data_root)
    if smoke:
        train = datasets.FakeData(2, (channels, size, size), classes, train_transform)
        test = datasets.FakeData(2, (channels, size, size), classes, test_transform)
    elif dataset == 'tinyimagenet':
        if (root / 'tiny-imagenet-200').exists():
            root = root / 'tiny-imagenet-200'
        if download and not (root / 'train').exists():
            download_tinyimagenet(str(root.parent if root.name == 'tiny-imagenet-200' else root))
            if (root / 'tiny-imagenet-200').exists():
                root = root / 'tiny-imagenet-200'
        train = datasets.ImageFolder(root=root / 'train', transform=train_transform)
        val = root / 'val'
        if (val / 'val_annotations.txt').exists() and any((val / 'images').glob('*.JPEG')):
            test = _TinyImageNetValidation(val, train.class_to_idx, test_transform)
        else:
            test = datasets.ImageFolder(root=val / 'images' if (val / 'images').exists() else val,
                                        transform=test_transform)
            if train.class_to_idx != test.class_to_idx:
                raise ValueError('TinyImageNet class mappings differ')
    elif dataset == 'svhn':
        train = datasets.SVHN(root=root, split='train', download=download, transform=train_transform)
        test = datasets.SVHN(root=root, split='test', download=download, transform=test_transform)
    else:
        constructor = datasets.MNIST if dataset == 'mnist' else datasets.CIFAR10
        train = constructor(root=root, train=True, download=download, transform=train_transform)
        test = constructor(root=root, train=False, download=download, transform=test_transform)
    selection = None
    if selection_split == 'validation' or (selection_split is None and val_split):
        evaluation = copy.copy(train)
        evaluation.transform = test_transform
        generator = None if seed is None else torch.Generator().manual_seed(seed)
        if selection_split is None:
            train, selection = random_split(train, [1 - validation_fraction, validation_fraction],
                                            generator=generator)
            selection = Subset(evaluation, selection.indices)
        else:
            indices = torch.randperm(len(train), generator=generator)
            count = max(1, int(len(train) * validation_fraction))
            if count >= len(train):
                raise ValueError('Holdout leaves no training samples')
            selection = Subset(evaluation, indices[:count].tolist())
            train = Subset(train, indices[count:].tolist())
    elif selection_split == 'test':
        selection = test

    def loader(data, shuffle=False, validation=False):
        num_workers = workers if workers is not None else (
            4 if dataset == 'tinyimagenet' and not validation else 2)
        result = DataLoader(data, batch_size=2 if smoke else batch_size,
                            shuffle=shuffle, num_workers=num_workers)
        result.mean, result.std, result.normalised = mean, std, normalise
        result.min = (-mean / std).view(1, channels, 1, 1)
        result.max = ((1 - mean) / std).view(1, channels, 1, 1)
        return result

    train_loader, test_loader = loader(train, True), loader(test)
    if selection_split is None and not val_split:
        return train_loader, test_loader
    selection_loader = None if selection is None else (
        test_loader if selection is test else loader(selection, validation=True))
    return train_loader, selection_loader, test_loader


def load_mnist(batch_size=64, normalise=True, train_transforms=(),
                  val_split=True, data_root='./data', *, workers=None, download=True,
                  selection_split=None, validation_fraction=.2, seed=None, smoke=False,
                  input_size=None, normalisation_stats=None):
    """
    Loads the MNIST dataset with specified transformations and returns data loaders.
    
    Args:
        batch_size (int, optional): The number of samples per batch to load. Default is 64.
        normalise (bool, optional): Whether to normalize the dataset. Default is True.
        train_transforms (list, optional): List of additional transformations to apply to the training data. Default is an empty list.
        val_split (bool, optional): Whether to split the training data into training and validation sets. Default is True.
        data_root (str, optional): Root directory where the dataset will be stored. Default is './data'.
    
    Returns:
        tuple: If val_split is True, returns a tuple of DataLoader objects (train_loader, val_loader, test_loader). Otherwise, returns (train_loader, test_loader).

    
    The returned DataLoader objects have the following additional attributes:
        - mean (torch.Tensor): The mean value used for normalization.
        - std (torch.Tensor): The standard deviation value used for normalization.
        - min (torch.Tensor): The minimum value in the dataset.
        - max (torch.Tensor): The maximum value in the dataset.
        - normalised (bool): Whether the dataset was normalized.
    
    Example:
        train_loader, val_loader, test_loader = load_mnist(batch_size=32, normalise=True, train_transforms=[transforms.RandomRotation(10)], val_split=True, data_root='./data')
    """
    return _load_image_data('mnist', batch_size, normalise, train_transforms,
                            val_split, data_root, workers, download, selection_split,
                            validation_fraction, seed, smoke, input_size, normalisation_stats)


def load_cifar10(batch_size=64, normalise=True, train_transforms=(transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 2, padding_mode='edge')),
                  val_split=True, data_root='./data', *, workers=None, download=True,
                  selection_split=None, validation_fraction=.2, seed=None, smoke=False,
                  input_size=None, normalisation_stats=None):
    """
    Load the CIFAR-10 dataset with optional normalization and data augmentation.
    
    Args:
        batch_size (int, optional): The number of samples per batch to load. Default is 64.
        normalise (bool, optional): Whether to normalize the dataset. Default is True.
        train_transforms (list, optional): List of transformations to apply to the training data. Default is [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 2, padding_mode='edge')].
        val_split (bool, optional): Whether to split the training data into training and validation sets. Default is True.
        data_root (str, optional): Root directory where the dataset will be stored. Default is './data'.
    
    Returns:
        tuple: If val_split is True, returns a tuple of DataLoader objects (train_loader, val_loader, test_loader). Otherwise, returns (train_loader, test_loader).
    
    The DataLoader objects have the following attributes:
        - mean (torch.Tensor): The mean used for normalization.
        - std (torch.Tensor): The standard deviation used for normalization.
        - min (torch.Tensor): The minimum value in the dataset.
        - max (torch.Tensor): The maximum value in the dataset.
        - normalised (bool): Whether the dataset was normalized.
    """
    return _load_image_data('cifar10', batch_size, normalise, train_transforms,
                            val_split, data_root, workers, download, selection_split,
                            validation_fraction, seed, smoke, input_size, normalisation_stats)


def load_svhn(batch_size=64, normalise=True, train_transforms=(),
                  val_split=True, data_root='./data', *, workers=None, download=True,
                  selection_split=None, validation_fraction=.2, seed=None, smoke=False,
                  input_size=None, normalisation_stats=None):
    """Load SVHN and attach the metadata required for bound propagation.

    Training transformations apply only to training images. A training holdout
    uses evaluation transformations, so validation does not augment its inputs.

    Args:
        batch_size (int): Number of images per loader batch.
        normalise (bool): Whether to apply channelwise dataset normalisation.
        train_transforms (Sequence): Image transformations preceding tensor conversion on
            training data.
        val_split (bool): Whether to create a training holdout when selection_split is None.
        data_root (str): Dataset storage directory.
        workers (int, optional): Number of loader workers; None uses the dataset default.
        download (bool): Whether to download missing data.
        selection_split (str, optional): 'validation' uses a training holdout, 'test' reuses
            the test loader, and 'none' disables selection.
            None follows val_split.
        validation_fraction (float): Training fraction reserved for a holdout, in (0, 1).
        seed (int, optional): Seed for selecting holdout indices.
        smoke (bool): Whether to return two synthetic images per split for execution checks.
        input_size (int, optional): Positive centre-crop size; None retains the dataset
            resolution.
        normalisation_stats (dict, optional): Custom channel mean and std with
            normalise=True.

    Returns:
        (tuple): (train_loader, test_loader) when selection_split is None and
            val_split=False; otherwise (train_loader, selection_loader, test_loader).
            The selection loader is None when selection is disabled. All loaders
            expose mean, std, min, max and normalised attributes.
    """
    return _load_image_data('svhn', batch_size, normalise, train_transforms,
                            val_split, data_root, workers, download, selection_split,
                            validation_fraction, seed, smoke, input_size, normalisation_stats)


def load_gtsrb(batch_size=64, normalise=True, train_transforms=[transforms.RandomRotation(10), transforms.RandomResizedCrop(size=(30, 30), scale=(0.85, 1.0)), transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=15),], val_split=True, data_root='./data'):
    """
    Load the German Traffic Sign Recognition Benchmark (GTSRB) dataset with optional normalization and data augmentation.
    
    Args:
        batch_size (int, optional): The number of samples per batch to load. Default is 64.
        normalise (bool, optional): Whether to normalize the dataset using predefined mean and std values. Default is True.
        train_transforms (list, optional): List of torchvision transforms to apply to the training data. Default includes random rotation, resized crop, and affine transformations.
        val_split (bool, optional): Whether to split the training data into training and validation sets. Default is True.
        data_root (str, optional): Root directory where the dataset is stored or will be downloaded. Default is './data'.
    
    Returns:
        tuple: If val_split is True, returns a tuple of DataLoader objects (train_loader, val_loader, test_loader). Otherwise, returns (train_loader, test_loader).
    
    The DataLoader objects returned have additional attributes:
        - mean (torch.Tensor): The mean used for normalization.
        - std (torch.Tensor): The standard deviation used for normalization.
        - min (torch.Tensor): The minimum value in the dataset.
        - max (torch.Tensor): The maximum value in the dataset.
        - normalised (bool): Whether the dataset was normalized.
    """
    if normalise:
        mean = torch.tensor([0.3403, 0.3121, 0.3214])
        std =  torch.tensor([0.2724, 0.2608, 0.2669])
    else:
        mean = torch.tensor([0, 0, 0])
        std = torch.tensor([1, 1, 1])
    if normalise:
        train_transform = transforms.Compose([
            *train_transforms,
            transforms.Resize((30, 30)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((30, 30)),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            *train_transforms,
            transforms.Resize((30, 30)),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((30, 30)),
            transforms.ToTensor(),
        ])

    train_dataset_ori = datasets.GTSRB(root=data_root, split='train', download=True, transform=train_transform)
    test_dataset = datasets.GTSRB(root=data_root, split='test', download=True, transform=test_transform)
    
    if val_split:
        file_names = train_dataset_ori._samples
        track_id_to_sample = [name[0].split("/")[-2] + "_" + name[0].split("/")[-1].split("_")[0] for name in file_names]
        track_ids = set(track_id_to_sample)
        train_track_ids, val_track_ids = train_test_split(np.array(sorted(track_ids)), test_size=0.2, train_size=0.8)
        train_ids = [i for i in range(len(track_id_to_sample)) if track_id_to_sample[i] in train_track_ids]
        val_ids = [i for i in range(len(track_id_to_sample)) if track_id_to_sample[i] in val_track_ids]

        
        train_dataset = Subset(train_dataset_ori, train_ids)
        evaluation_dataset = copy.copy(train_dataset_ori)
        evaluation_dataset.transform = test_transform
        val_dataset = Subset(evaluation_dataset, val_ids)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        val_loader.mean, val_loader.std = mean, std

    else:
        train_dataset = train_dataset_ori


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize variables to track the min and max values
    min_val = ((torch.zeros(mean.shape) - mean) / std).view(1, *mean.shape, 1, 1)
    max_val = ((torch.ones(mean.shape) - mean) / std).view(1, *mean.shape, 1, 1)

    max_target = float('-inf')
    # Loop over the dataset
    for data, target in train_loader:
        max_target = torch.tensor(max(max_target, target.max().item()))
        

    print(f'GTSRB dataset - Min value: {min_val}, Max value: {max_val}, no_classes: {max_target}')

    train_loader.mean, train_loader.std, train_loader.min, train_loader.max, train_loader.normalised = mean, std, min_val, max_val, normalise
    test_loader.mean, test_loader.std, test_loader.min, test_loader.max, test_loader.normalised = mean, std, min_val, max_val, normalise

    if val_split:
        val_loader.mean, val_loader.std, val_loader.min, val_loader.max, val_loader.normalised = mean, std, min_val, max_val, normalise
        return train_loader, val_loader, test_loader
    else:
        return train_loader, test_loader
    
def download_tinyimagenet(data_root='./data'):
    """
    Downloads the TinyImageNet dataset and extracts it to the specified directory.
    
    Args:
        data_root (str, optional): Root directory where the dataset will be stored. Default is './data'.
    """
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    dataset_dir = os.path.join(data_root, 'tiny-imagenet-200')
    zip_path = os.path.join(data_root, 'tiny-imagenet-200.zip')

    # Check if the dataset directory exists and contains the expected subdirectories
    expected_subdirs = ['train', 'val', 'test', 'wnids.txt', 'words.txt']
    if not os.path.exists(dataset_dir) or not all(os.path.exists(os.path.join(dataset_dir, subdir)) for subdir in expected_subdirs):
        os.makedirs(data_root, exist_ok=True)
        print(f'Downloading TinyImageNet dataset from {url}...')
        response = requests.get(url, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print('Extracting TinyImageNet dataset...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_root)
        os.remove(zip_path)
        print('TinyImageNet dataset downloaded and extracted successfully.')
        create_val_img_folder(dataset_dir)
        print('Validation images separated into subfolders.')
    else:
        print('TinyImageNet dataset already exists and is complete.')




def load_tinyimagenet(batch_size=64, normalise=True, train_transforms=(transforms.RandomHorizontalFlip(), transforms.RandomCrop(64, 4, padding_mode='edge')),
                  val_split=True, data_root='./data', *, workers=None, download=True,
                  selection_split=None, validation_fraction=.2, seed=None, smoke=False,
                  input_size=None, normalisation_stats=None):
    """
    Loads the TinyImageNet dataset with specified transformations and normalization.
    
    Args:
        batch_size (int, optional): The number of samples per batch. Default is 64.
        normalise (bool, optional): Whether to normalize the dataset. Default is True.
        train_transforms (list, optional): List of transformations to apply to the training data. Default is [transforms.RandomHorizontalFlip(), transforms.RandomCrop(64, 4, padding_mode='edge')].
        val_split (bool, optional): Whether to split the training data into training and validation sets. Default is True.
        data_root (str, optional): Root directory of the dataset. Default is './data'.
    
    Returns:
        tuple: If val_split is True, returns a tuple of DataLoader objects (train_loader, val_loader, test_loader). Otherwise, returns (train_loader, test_loader).
    
    The returned DataLoader objects have the following attributes:
        - mean: The mean used for normalization.
        - std: The standard deviation used for normalization.
        - min: The minimum value in the dataset.
        - max: The maximum value in the dataset.
        - normalised: Whether the dataset was normalized.
    """
    return _load_image_data('tinyimagenet', batch_size, normalise, train_transforms,
                            val_split, data_root, workers, download, selection_split,
                            validation_fraction, seed, smoke, input_size, normalisation_stats)
