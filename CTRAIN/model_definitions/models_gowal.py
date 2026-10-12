import torch.nn as nn



class GowalConvSmall(nn.Module):
    """
    GowalConvSmall is a small convolutional neural network model designed for certified training, proposed by Gowal et al.
    
    Args:
        in_shape (tuple): Shape of the input tensor. Default is (1, 28, 28) for MNIST dataset.
        n_classes (int): Number of output classes. Default is 10.
        dataset (str): The dataset being used. Can be 'mnist' or 'cifar10'. Default is 'mnist'.
    
    Attributes:
        layers (nn.Sequential): A sequential container of the layers in the network.
    
    Methods:
        forward(x):
            Defines the forward pass of the network.
    """
    def __init__(self, in_shape=(1, 28, 28), n_classes=10, dataset='mnist'):
        super(GowalConvSmall, self).__init__()
        assert in_shape[1] == in_shape[2], "We only support square inputs for now!"
        in_channels = in_shape[0]
        in_dim  = in_shape[1]
        if dataset == 'mnist':
            linear_in = 3200
        elif dataset == 'cifar10':
            linear_in = 4608
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(4,4), stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(linear_in, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, n_classes)
        )
    
    def forward(self, x):
        return self.layers(x)


class GowalConvMed(nn.Module):
    """
    A convolutional medium-sized neural network model based on the architecture proposed by Gowal et al.
    
    Args:
        in_shape (tuple): Shape of the input tensor. Default is (1, 28, 28) for MNIST dataset.
        n_classes (int): Number of output classes. Default is 10.
        dataset (str): The dataset being used. Can be 'mnist' or 'cifar10'. Default is 'mnist'.
    
    Attributes:
        layers (nn.Sequential): A sequential container of the layers in the network.
    
    Methods:
        forward(x):
            Defines the forward pass of the network.
    """
    def __init__(self, in_shape=(1, 28, 28), n_classes=10, dataset='mnist'):
        super(GowalConvMed, self).__init__()
        in_channels = in_shape[0]
        if dataset == 'mnist':
            linear_in = 1024
        elif dataset == 'cifar10':
            linear_in = 1600
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(4, 4), stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(linear_in, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class GowalConvLarge(nn.Module):
    """
    A large convolutional neural network model based on the architecture proposed by Gowal et al.
    
    Args:
        in_shape (tuple): Shape of the input tensor. Default is (1, 28, 28) for MNIST dataset.
        n_classes (int): Number of output classes. Default is 10.
        dataset (str): The dataset being used. Can be 'mnist' or 'cifar10'. Default is 'mnist'.
    
    Attributes:
        layers (nn.Sequential): A sequential container of the layers in the network.
    
    Methods:
        forward(x):
            Defines the forward pass of the network.
    """
    def __init__(self, in_shape=(1, 28, 28), n_classes=10, dataset='mnist'):
        super(GowalConvLarge, self).__init__()
        in_channels = in_shape[0]
        if dataset == 'mnist':
            linear_in = 6272
        elif dataset == 'cifar10':
            linear_in = 10368
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3,3), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(linear_in, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_classes)
        )
    
    def forward(self, x):
        return self.layers(x)