"""Forward LiRPA evaluation bounds for complete clipped input boxes."""
import torch
from auto_LiRPA import BoundedTensor
from auto_LiRPA.operators import BoundRelu
from CTRAIN.util import construct_c


def bound_forward(model, ptb, data, target, n_classes=10,
                  bound_upper=False, reuse_input=False):
    """Evaluation-only same-slope margins; restore caller relaxation options."""
    relaxations = {node: node.relu_options for node in model.modules()
                   if isinstance(node, BoundRelu)}
    try:
        for node in relaxations:
            node.relu_options = 'same-slope'
        with torch.no_grad():
            return model.compute_bounds(
                x=None if reuse_input else (BoundedTensor(data, ptb),),
                C=construct_c(data, target, n_classes), method='Forward',
                bound_upper=bound_upper)
    finally:
        for node, option in relaxations.items():
            node.relu_options = option
