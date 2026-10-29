"""Certified classification objectives using CROWN-IBP and optional loss fusion."""
import torch
from auto_LiRPA.bound_ops import BoundExp

from CTRAIN.bound import bound_ibp, bound_crown_ibp
from CTRAIN.train.certified.losses import get_loss_from_bounds


def get_crown_ibp_loss(hardened_model, ptb, data, target, n_classes, criterion, beta, loss_fusion=False, return_bounds=False, return_stats=True):
    """Compute a classification loss from mixed CROWN-IBP and IBP bounds.

    The parameter beta weights CROWN-IBP bounds against IBP bounds. For loss
    fusion, first evaluate hardened_model(data, target) on the current batch:
    auto_LiRPA's fused backward pass requires the current inputs and labels.
    The fused path returns NaN for the unavailable robust-error statistic.

    Args:
        hardened_model (auto_LiRPA.BoundedModule): The bounded model to be trained.
        ptb (auto_LiRPA.PerturbationLpNorm): The perturbation applied to the input data.
        data (torch.Tensor): The input data.
        target (torch.Tensor): The target labels.
        n_classes (int): The number of classes in the classification task.
        criterion (callable): The loss function to be used.
        beta (float): The interpolation parameter between CROWN_IBP and IBP bounds.
        loss_fusion (bool, optional): If True, use loss fusion.
        return_bounds (bool, optional): If True, return the lower bounds.
        return_stats (bool, optional): If True, return the robust error statistics.

    Returns:
        (tuple): The scalar loss, followed by (lb, ub) when return_bounds=True
            and the robust-error fraction when return_stats=True. With loss fusion,
            margin error is unavailable and the statistic is NaN.
    """
    ilb, iub = bound_ibp(
        model=hardened_model,
        ptb=ptb,
        data=data,
        target=target,
        n_classes=n_classes,
        bound_upper=True if loss_fusion else False,
        loss_fusion=loss_fusion,
    )
    lb, ub = ilb, iub
    if beta >= 1e-5:
        clb, cub = bound_crown_ibp(
            model=hardened_model,
            ptb=ptb,
            data=data,
            target=target,
            n_classes=n_classes,
            reuse_input=False,
            bound_upper=True if loss_fusion else False,
            loss_fusion=loss_fusion,
        )
        lb = clb * beta + ilb * (1 - beta) if clb is not None and ilb is not None else None
        ub = cub * beta + iub * (1 - beta) if cub is not None and iub is not None else None

    if loss_fusion:
        exp_module = get_exp_module(hardened_model)
        max_input = exp_module.max_input
        certified_loss = torch.mean(torch.log(ub) + max_input)
    else:
        certified_loss = get_loss_from_bounds(lb, criterion)
        
    return_tuple = (certified_loss,)
    
    if return_bounds:
        return_tuple = return_tuple + (lb, ub)
    if return_stats:
        robust_err = torch.sum(~(lb > 0).all(dim=1)).item() / data.size(0) if not loss_fusion else float('nan')
        return_tuple = return_tuple + (robust_err,)
    
    return return_tuple


def get_exp_module(bounded_module):
    """Find the exponential node required to recover a fused cross-entropy bound; return None
    if absent.
    """
    for _, node in bounded_module.named_modules():
        # Find the Exp neuron in computational graph
        if isinstance(node, BoundExp):
            return node
    return None
