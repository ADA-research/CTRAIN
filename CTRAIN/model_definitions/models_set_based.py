"""Convolutional architecture used for CTRAIN's set-based training experiments."""
from torch import nn


class CNN6_SetBased(nn.Module):
    """Construct the Cnn6 architecture used for set-based training.

    The model combines four convolutional layers with BatchNorm and ReLU,
    followed by three affine layers. Its layout follows the interpretation of
    Appendix A of Koller et al. (2025) used by CTRAIN.

    Args:
        in_shape (tuple[int, ...]): Input shape (channels, height, width), excluding the batch dimension.
        n_classes (int): Number of output classes.
    """
    def __init__(self, in_shape=(1, 28, 28), n_classes=10):
        super().__init__()
        channels, height, width = in_shape
        layers = []
        for output, kernel, stride in [(32, 3, 1), (32, 4, 2), (64, 3, 1), (64, 4, 2)]:
            layers.extend([nn.Conv2d(channels, output, kernel, stride, padding=1),
                           nn.BatchNorm2d(output), nn.ReLU()])
            height = (height + 2 - kernel) // stride + 1
            width = (width + 2 - kernel) // stride + 1
            channels = output
        if min(height, width) <= 0:
            raise ValueError('Input spatial size too small for Cnn6')
        layers.extend([nn.Flatten(), nn.Linear(channels * height * width, 512),
                       nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 512),
                       nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, n_classes)])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Return network predictions for the supplied input batch."""
        return self.layers(x)
