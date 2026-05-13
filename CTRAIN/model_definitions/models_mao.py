import torch.nn as nn


class CNN3_Mao(nn.Module):
    def __init__(self, in_shape=(1, 28, 28), width=64, linear_size=512, n_classes=10):
        super(CNN3_Mao, self).__init__()
        assert in_shape[1] == in_shape[2], "We only support square inputs for now!"
        in_channels = in_shape[0]
        in_dim = in_shape[1]

        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels, width, 5, stride=2, padding=2),
                nn.BatchNorm2d(width),
                nn.ReLU(),
                nn.Conv2d(width, 2 * width, 4, stride=2, padding=1),
                nn.BatchNorm2d(2 * width),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear((in_dim // 4) ** 2 * 2 * width, n_classes),
            )
        )

    def forward(self, x):
        return self.layers(x)


class CNN5_Mao(nn.Module):
    def __init__(self, in_shape=(1, 28, 28), width=64, linear_size=512, n_classes=10):
        super(CNN5_Mao, self).__init__()
        assert in_shape[1] == in_shape[2], "We only support square inputs for now!"
        in_channels = in_shape[0]
        in_dim = in_shape[1]

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, 4, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, 2 * width, 4, stride=2, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((in_dim // 4) ** 2 * 2 * width, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Linear(linear_size, n_classes),
        )

    def forward(self, x):
        return self.layers(x)


class CNN9_Mao(nn.Module):
    def __init__(self, in_shape=(1, 28, 28), width=64, linear_size=512, n_classes=10):
        super(CNN9_Mao, self).__init__()
        assert in_shape[1] == in_shape[2], "We only support square inputs for now!"
        in_channels = in_shape[0]
        in_dim = in_shape[1]

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((in_dim // 2) * (in_dim // 2) * 2 * width, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Linear(linear_size, n_classes),
        )

    def forward(self, x):
        return self.layers(x)


class CNN11_Mao(nn.Module):
    def __init__(self, in_shape=(1, 28, 28), width=64, linear_size=512, n_classes=10):
        super(CNN11_Mao, self).__init__()
        assert in_shape[1] == in_shape[2], "We only support square inputs for now!"
        in_channels = in_shape[0]
        in_dim = in_shape[1]

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((in_dim // 2) * (in_dim // 2) * 2 * width, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Linear(linear_size, n_classes),
        )

    def forward(self, x):
        return self.layers(x)
