"""CNN3/CNN5 architectures from Table 9 of arXiv:2403.07095."""
import torch
from torch import nn


class CNN_GLS(nn.Sequential):
    """Paper CNNs without BatchNorm or an embedded normalization layer.

    ``depth=3, width=8`` gives CNN3; width=2 gives CNN3Tiny.
    ``depth=5, width=16`` gives CNN5; width=64 gives CNN5L.
    Inputs must already be normalized by the data loader.
    """
    def __init__(self, in_shape=(1, 28, 28), n_classes=10, depth=5,
                 width=16, linear_size=512, initialization='default'):
        if depth not in (3, 5) or width < 1:
            raise ValueError('depth must be 3 or 5 and width must be positive')
        if initialization not in ('default', 'kaiming_uniform'):
            raise ValueError('unsupported GLS initialization')
        layers = [nn.Conv2d(in_shape[0], width, 5, stride=2, padding=2), nn.ReLU(),
                  nn.Conv2d(width, width, 4, stride=2, padding=1), nn.ReLU()]
        if depth == 5:
            layers += [nn.Conv2d(width, 2 * width, 4, stride=2, padding=1), nn.ReLU()]
        with torch.no_grad():
            flat_size = nn.Sequential(*layers)(torch.zeros(1, *in_shape)).numel()
        layers += [nn.Flatten()]
        if depth == 5:
            layers += [nn.Linear(flat_size, linear_size), nn.ReLU()]
            flat_size = linear_size
        layers += [nn.Linear(flat_size, n_classes)]
        super().__init__(*layers)
        if initialization == 'kaiming_uniform':
            for layer in self:
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_uniform_(layer.weight)
