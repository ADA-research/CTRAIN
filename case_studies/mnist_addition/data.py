"""Normalized MNIST pairs with optional complete coverage in both positions."""
import math
import hashlib

import torch
from torch.utils.data import TensorDataset
from torchvision.transforms import Normalize
from CTRAIN.data_loaders.data_loaders import MNIST_MEAN, MNIST_STD
from CTRAIN.properties import Box


class MNISTPropertyDataset:
    """Lazy box around every normalized training pair; no endpoint/file pool."""
    def __init__(self, dataset, epsilon, count=None):
        if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError('epsilon must be in [0,1]')
        if count is not None and (type(count) is not int or not 1 <= count <= len(dataset)):
            raise ValueError('Box count must be between 1 and the number of training pairs')
        self.inputs = dataset.tensors[0][:count]
        if (not len(self.inputs) or tuple(self.inputs.shape[1:]) != (2, 28, 28)
                or self.inputs.device.type != 'cpu' or not self.inputs.is_contiguous()):
            raise ValueError('Expected nonempty contiguous CPU MNIST pair inputs')
        self.epsilon = epsilon
        self.inputs_sha256 = hashlib.sha256(memoryview(self.inputs.numpy())).hexdigest()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        normalized = self.inputs[index].double()
        if self.epsilon == 0:
            # Use the represented model input exactly, without inverse-normalization rounding.
            return Box(normalized, normalized)
        # Recover original uint8 pixels instead of retaining normalization error.
        raw = ((normalized * MNIST_STD + MNIST_MEAN) * 255).round() / 255
        lower = ((raw - self.epsilon).clamp(0, 1) - MNIST_MEAN) / MNIST_STD
        upper = ((raw + self.epsilon).clamp(0, 1) - MNIST_MEAN) / MNIST_STD
        return Box(lower, upper)

    def to_dict(self):
        return dict(kind='mnist_pair_points' if self.epsilon == 0 else 'mnist_pair_pixel_boxes',
                    count=len(self), epsilon=self.epsilon,
                    inputs_sha256=self.inputs_sha256, input_shape=[2, 28, 28],
                    normalization=dict(mean=MNIST_MEAN, std=MNIST_STD),
                    center_reconstruction='normalized input exactly' if self.epsilon == 0 else
                                          'round((normalized*std+mean)*255)/255',
                    sampling='uniform with replacement')


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
