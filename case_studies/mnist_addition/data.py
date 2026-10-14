"""Normalized MNIST pairs with optional complete coverage in both positions."""
import math

import torch
from torch.utils.data import TensorDataset
from torchvision.transforms import Normalize
from CTRAIN.data_loaders.data_loaders import MNIST_MEAN, MNIST_STD


def paired_mnist_dataset(images, labels, count, seed, *, cover_all=False):
    """Return normalized image pairs, integer [a,b,a+b] targets and source indices.

    With ``cover_all=True``, each position contains every source image before
    starting another independently shuffled pass. The requested count must
    cover at least one full pass. Otherwise pairs are sampled with replacement.
    """
    if (images.dtype != torch.uint8 or labels.dtype != torch.long
            or images.shape != (len(labels), 28, 28) or labels.ndim != 1
            or not len(labels) or type(count) is not int or count < 1
            or not ((labels >= 0) & (labels <= 9)).all()):
        raise ValueError('Expected nonempty uint8 MNIST images, long labels 0–9 and a positive pair count')
    if type(cover_all) is not bool or (cover_all and count < len(labels)):
        raise ValueError('Complete coverage requires at least as many pairs as source images')
    generator = torch.Generator().manual_seed(seed)
    if cover_all:
        indices = torch.stack([
            torch.cat([torch.randperm(len(labels), generator=generator)
                       for _ in range(math.ceil(count / len(labels)))])[:count]
            for _ in range(2)], dim=1)
    else:
        indices = torch.randint(len(labels), (count, 2), generator=generator)
    operands = labels[indices]
    targets = torch.cat((operands, operands.sum(dim=1, keepdim=True)), dim=1)
    inputs = Normalize([MNIST_MEAN] * 2, [MNIST_STD] * 2)(images[indices].float() / 255)
    return TensorDataset(inputs, targets), indices
