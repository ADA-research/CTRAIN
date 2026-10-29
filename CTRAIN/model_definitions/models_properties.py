"""Task models and composed verification graphs for general input-output properties."""
import torch
from torch import nn


class MNISTAdditionRegressor(nn.Sequential):
    """Predict two MNIST digits and their sum through independent scalar outputs.

    A two-hidden-layer MLP receives the concatenated images and returns [a, b,
    sum]. No output is computed from another, so arithmetic consistency is a
    property to be learned and verified.

    Args:
        width (int): Number of units or base channels in the hidden layers.
    """
    def __init__(self, width=64):
        if type(width) is not int or width < 1:
            raise ValueError('width must be a positive integer')
        super().__init__(nn.Flatten(), nn.Linear(2 * 28 * 28, width), nn.ReLU(),
                         nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 3))


class MNISTAdditionClassifier(nn.Sequential):
    """Predict two MNIST digits and their sum using three independent heads.

    The two-hidden-layer MLP returns 39 logits: ten for each operand and nineteen
    for their sum. The sum head is learned independently, so task accuracy and
    arithmetic consistency can be trained and assessed separately.

    Args:
        width (int): Number of units or base channels in the hidden layers.
    """
    head_sizes = (10, 10, 19)

    def __init__(self, width=64):
        if type(width) is not int or width < 1:
            raise ValueError('width must be a positive integer')
        super().__init__(nn.Flatten(), nn.Linear(2 * 28 * 28, width), nn.ReLU(),
                         nn.Linear(width, width), nn.ReLU(), nn.Linear(width, sum(self.head_sizes)))


class MNISTSwapGraph(nn.Module):
    """Evaluate both operand orders through one shared MNIST addition classifier.

    The graph concatenates the original and swapped predictions into 78 logits.
    This exposes relational properties while sharing all classifier parameters.

    Args:
        model (torch.nn.Module): PyTorch model to train or evaluate.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        """Return network predictions for the supplied input batch."""
        swapped = torch.cat((inputs[:, 1:2], inputs[:, 0:1]), dim=1)
        return torch.cat((self.model(inputs), self.model(swapped)), dim=1)


class ControllerMLP(nn.Sequential):
    """Construct a two-input, two-output controller with two 32-unit ReLU layers.

    Optional coordinate-feature initialisation exposes signed state features and
    starts with zero actions. All layers remain trainable; this initialisation
    does not itself ensure invariance under disturbances.

    Args:
        coordinate_features (bool): Whether to initialise signed coordinate features with zero
            output actions.
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
    """Compose a controller with fixed additive state-transition dynamics.

    Inputs contain [x0, x1, w0, w1]. Outputs contain the two controller actions
    followed by successor states x_next = x + u + w. This graph lets output
    specifications constrain actions and next states with shared controller weights.

    Args:
        controller (torch.nn.Module): Controller mapping the two state coordinates to two actions.
    """
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def forward(self, inputs):
        """Return network predictions for the supplied input batch."""
        action = self.controller(inputs[:, :2])
        successor = inputs[:, :2] + action + inputs[:, 2:]
        return torch.cat((action, successor), dim=1)
