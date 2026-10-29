"""Weight initialisation for stable IBP training following Shi et al. (2021)."""
import torch
import math
from auto_LiRPA.operators import BoundConv, BoundLinear, BoundAdd

def get_params(model):
    """Collect named weight and bias parameter pairs for Shi initialisation."""
    weights = []
    biases = []
    for p in model.named_parameters():
        if 'weight' in p[0]:
            weights.append(p)
        elif 'bias' in p[0]:
            biases.append(p)
        else:
            print('Skipping parameter {}'.format(p[0]))
    return weights, biases

def ibp_init_shi(model_ori, model, init_last_layer=False, zero_bias=False):
    """Initialise weights for stable IBP training following Shi et al. (2021).

    Affine and convolutional weights use standard deviation sqrt(2*pi)/fan_in.
    By default the final named weight tensor and existing biases are retained;
    init_last_layer and zero_bias explicitly change those choices. Residual
    connections receive the corresponding scale adjustment in the bounded graph.

    Args:
        model_ori (torch.nn.Module): The original model.
        model (auto_LiRPA.BoundedModule): The LiRPA model.
        init_last_layer (bool): Also reinitialize the final weight tensor.
            False preserves the original Shi code's exclusion.
        zero_bias (bool): Zero all named bias parameters, including BN offsets.
            False preserves their existing values.

    Returns:
        (None): This function updates the supplied state in place.
    """
    if type(init_last_layer) is not bool or type(zero_bias) is not bool:
        raise ValueError('init_last_layer and zero_bias must be booleans')
    weights, biases = get_params(model_ori)
    for i in range(len(weights) if init_last_layer else len(weights)-1):
        if weights[i][1].ndim == 1:
            continue
        weight = weights[i][1]
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        std = math.sqrt(2 * math.pi / (fan_in**2))     
        std_before = weight.std().item()
        torch.nn.init.normal_(weight, mean=0, std=std)
        print(f'Reinitialize {weights[i][0]}, std before {std_before:.5f}, std now {weight.std():.5f}')
    if zero_bias:
        for _, bias in biases:
            torch.nn.init.zeros_(bias)
    for node in model._modules.values():
        if isinstance(node, BoundConv) or isinstance(node, BoundLinear):
            if len(node.inputs[0].inputs) > 0 and isinstance(node.inputs[0].inputs[0], BoundAdd):
                print(f'Adjust weights for node {node.name} due to residual connection')
                node.inputs[1].param.data /= 2
