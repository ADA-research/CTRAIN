import torch.nn as nn

class CNN7_Shi(nn.Module):
    """
    CNN7_Shi is a convolutional neural network model designed for image classification tasks proposed by Shi et al.
    It is the defacto standard neural network architecture to evaluate certified training methods on.
    
    Args:
        in_shape (tuple): Shape of the input images. Default is (1, 28, 28).
        width (int): Number of channels for the convolutional layers. Default is 64.
        linear_size (int): Size of the fully connected layer. Default is 512.
        n_classes (int): Number of output classes. Default is 10.
        downsample_final (bool): Use stride 2 in the final convolution (TinyImageNet CNN7).
    
    Attributes:
        layers (nn.Sequential): Sequential container of the layers in the network.
    
    Methods:
        forward(x):
            Defines the forward pass of the network.
    """
    def __init__(self, in_shape=(1, 28, 28), width=64, linear_size=512, n_classes=10, downsample_final=False):
        super(CNN7_Shi, self).__init__()
        assert in_shape[1] == in_shape[2], "We only support square inputs for now!"
        in_channels = in_shape[0]
        in_dim  = in_shape[1]
        final_stride = 2 if downsample_final else 1
        feature_dim = in_dim // (2 * final_stride)
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, 2 * width, kernel_size=(3,3), stride=2, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, kernel_size=(3,3), stride=final_stride, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(feature_dim * feature_dim * 2 * width, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Linear(linear_size, n_classes)
        )
    
    def forward(self, x):
        return self.layers(x)