import torch
import torch.nn as nn
import numpy as np


def split_network(model, block_sizes, network_input, device):
    """
    Splits a neural network model into smaller sequential blocks based on specified block sizes. Needed for TAPS/STAPS.
    Args:
        model (torch.nn.Module): The neural network model to be split.
        block_sizes (list of int): A list of integers specifying the sizes of each block.
        network_input (torch.Tensor): The input tensor to the network.
        device (torch.device): The device to which the tensors should be moved (e.g., 'cpu' or 'cuda').
    Returns:
        list of torch.nn.Sequential: A list of sequential blocks representing the split network.
    """
    if len(block_sizes) != 2 or any(not isinstance(size, int) or size <= 0 for size in block_sizes):
        raise ValueError("TAPS/STAPS require two positive integer block sizes")
    if sum(block_sizes) != len(model.layers):
        raise ValueError("Block sizes must cover every network layer exactly once")
    start = 0
    original_blocks = []
    network_input = network_input.to(device)
    for size in block_sizes:
        end = start + size
        abs_block = nn.Sequential(model.layers[start:end])
        original_blocks.append(abs_block)
        
        output_shape = abs_block(network_input).shape
        network_input = torch.zeros(output_shape).to(device)
        
        start = end
    return original_blocks


def evaluate_taps_proxy(original_model, hardened_model, data_loader, eps, n_classes,
                        device, propagation="IBP", sabr_args=None, **attack_args):
    """Evaluate latent-margin accuracy without updating BatchNorm statistics."""
    from auto_LiRPA import PerturbationLpNorm
    from CTRAIN.bound import bound_taps
    from CTRAIN.util.util import preserve_model_state

    models = (original_model, hardened_model, *hardened_model.bounded_blocks)
    correct = total = 0
    with preserve_model_state(*models), torch.no_grad():
        for model in models:
            model.eval()
        radius = torch.as_tensor(eps, device=device).reshape(-1, 1, 1)
        data_min, data_max = data_loader.min.to(device), data_loader.max.to(device)
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            if (radius == 0).all():
                correct += (hardened_model(data).argmax(1) == target).sum().item()
            else:
                ptb = PerturbationLpNorm(
                    norm=np.inf, eps=radius,
                    x_L=torch.clamp(data - radius, data_min, data_max),
                    x_U=torch.clamp(data + radius, data_min, data_max),
                )
                region_args = None if sabr_args is None else dict(
                    sabr_args, hardened_model=hardened_model, original_model=original_model,
                    data=data, target=target, eps=radius, data_min=data_min,
                    data_max=data_max, device=device, n_classes=n_classes,
                )
                margins, _ = bound_taps(
                    original_model, hardened_model, hardened_model.bounded_blocks,
                    data, target, n_classes, ptb, device=device,
                    propagation=propagation, sabr_args=region_args, **attack_args,
                )
                correct += (margins > 0).all(1).sum().item()
            total += len(data)
    if not total:
        raise ValueError("Validation loader must contain at least one sample")
    return correct / total


def synchronise_bn(original_model, hardened_model, train_loader, device, population=False, loss_fusion_model=None):
    """Export current full-model buffers; optionally use CTBench population BN."""
    import torch
    from torch.nn.modules.batchnorm import _BatchNorm
    from CTRAIN.util.util import preserve_model_state

    original = original_model
    with torch.no_grad():
        buffers = dict(original.named_buffers())
        for name, value in hardened_model.named_buffers():
            buffers[name].copy_(value)
        layers = [m for m in original.modules() if isinstance(m, _BatchNorm)]
        if population and layers:
            momenta = [m.momentum for m in layers]
            try:
                with preserve_model_state(original):
                    original.train()
                    for batch, (data, _) in enumerate(train_loader, 1):
                        for layer in layers:
                            layer.momentum = 1 / batch
                        original(data.to(device))
            finally:
                for layer, momentum in zip(layers, momenta):
                    layer.momentum = momentum
        pairs = [(original, hardened_model),
                 *zip(getattr(hardened_model, "original_blocks", ()),
                      getattr(hardened_model, "bounded_blocks", ()))]
        for source, bounded in pairs:
            buffers = dict(source.named_buffers())
            for name, value in bounded.named_buffers():
                value.copy_(buffers[name])
        if loss_fusion_model is not None:
            buffers = dict(original.named_buffers())
            for name, value in loss_fusion_model.named_buffers():
                value.copy_(buffers[name.removeprefix("model.")])
