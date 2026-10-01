"""Appendix A Cnn6 interpretation, separate from CNN7_Shi."""
from torch import nn


class CNN6_SetBased(nn.Module):
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
        return self.layers(x)
