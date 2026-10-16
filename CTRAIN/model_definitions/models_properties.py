"""Small models and known dynamics for general-property examples."""
import torch
from torch import nn


class MNISTAdditionRegressor(nn.Sequential):
    """Two normalized MNIST images → independently learned [a, b, sum] scalars.

    The final affine layer has three unconstrained rows; no output is computed
    from another. Arithmetic consistency must be learned, not imposed here.
    """
    def __init__(self, width=64):
        if type(width) is not int or width < 1:
            raise ValueError('width must be a positive integer')
        super().__init__(nn.Flatten(), nn.Linear(2 * 28 * 28, width), nn.ReLU(),
                         nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 3))


class MNISTAdditionClassifier(nn.Sequential):
    """Independent logits for a (10), b (10), and a+b (19); unchanged MLP body.

    The sum head learns its own rows, supervised by the sum of ground-truth
    labels. Its output is never calculated from the operand predictions.
    """
    head_sizes = (10, 10, 19)

    def __init__(self, width=64):
        if type(width) is not int or width < 1:
            raise ValueError('width must be a positive integer')
        super().__init__(nn.Flatten(), nn.Linear(2 * 28 * 28, width), nn.ReLU(),
                         nn.Linear(width, width), nn.ReLU(), nn.Linear(width, sum(self.head_sizes)))


class MNISTSwapGraph(nn.Module):
    """One input pair, two operand orders, one shared classifier; 78 logits."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        swapped = torch.cat((inputs[:, 1:2], inputs[:, 0:1]), dim=1)
        return torch.cat((self.model(inputs), self.model(swapped)), dim=1)


class ControllerMLP(nn.Sequential):
    """2→32→32→2 ReLU controller, optionally initialized with coordinate features.

    The feature initialization has zero actions, hence is NOT robustly invariant
    under nonzero disturbances. All layers remain trainable.
    """
    def __init__(self, coordinate_features=False):
        super().__init__(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32),
                         nn.ReLU(), nn.Linear(32, 2))
        if coordinate_features:
            with torch.no_grad():
                for layer in (self[0], self[2], self[4]):
                    layer.weight.zero_()
                    layer.bias.zero_()
                for coordinate in range(2):
                    for j, (sign, offset) in enumerate(((1., 0.), (-1., 0.),
                                                       (1., -.3), (-1., -.3))):
                        row = coordinate * 4 + j
                        self[0].weight[row, coordinate] = sign
                        self[0].bias[row] = offset
                self[2].weight.copy_(torch.eye(32))


class ControllerDynamics(nn.Module):
    """Input [x0,x1,w0,w1]; output [u0,u1,x_next0,x_next1]."""
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def forward(self, inputs):
        action = self.controller(inputs[:, :2])
        successor = inputs[:, :2] + action + inputs[:, 2:]
        return torch.cat((action, successor), dim=1)
