"""Compute CROWN, CROWN-IBP and alpha-CROWN bounds through auto_LiRPA."""
import copy
import torch
from auto_LiRPA import BoundedTensor

from CTRAIN.util import construct_c
from CTRAIN.util.util import preserve_model_state


def bound_alpha_crown(model, ptb, data, target, n_classes=10,
                      bound_upper=False, reuse_input=False, alpha_iterations=20):
    """Compute classification-margin bounds by optimising CROWN relaxation slopes.

    Alpha-CROWN tightens linear relaxations through gradient-based slope
    optimisation. The computation enables the required gradients even inside
    no_grad, without accumulating parameter gradients. Caller modes, gradient
    flags and bound options are restored after the computation.
    """
    if type(alpha_iterations) is not int or alpha_iterations <= 0:
        raise ValueError('alpha_iterations must be a positive integer')
    options = copy.deepcopy(model.bound_opts)
    try:
        with preserve_model_state(model), torch.enable_grad():
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.set_bound_opts({'optimize_bound_args': {'iteration': alpha_iterations}})
            bounded = BoundedTensor(data, ptb=ptb)
            return model.compute_bounds(
                x=None if reuse_input else (bounded,), IBP=False,
                method='CROWN-Optimized', C=construct_c(data, target, n_classes),
                bound_upper=bound_upper)
    finally:
        model.set_bound_opts(options)

# Uses CROWN bounds throughout all intermediate layers and the final layer. 
def bound_crown(model, ptb, data, target, n_classes=10, bound_upper=False, reuse_input=False):
    """Compute classification-margin bounds using CROWN linear relaxations.

    CROWN uses linear relaxations throughout the network to bound each true-class
    logit minus its competing logits over the supplied input region.

    Args:
        model (auto_LiRPA.BoundedModule): The neural network model for which bounds are to
            be computed.
        ptb (auto_LiRPA.PerturbationLpNorm): The perturbation object defining the
            perturbation set.
        data (Tensor): The input data tensor.
        target (Tensor): The target labels tensor.
        n_classes (int, optional): The number of classes for classification.
        bound_upper (bool, optional): Whether to compute the upper bound.
        reuse_input (bool, optional): Whether to reuse the input data from previous bounding
            operation.

    Returns:
        (tuple): Lower and optional upper bounds on the requested outputs or
            margins. The upper entry is None when upper bounds are not computed.
    """
    data = BoundedTensor(data, ptb=ptb)
    c = construct_c(data, target, n_classes)
    if reuse_input:
        bound_input = None
    else:
        bound_input = (data,)
    lb, ub = model.compute_bounds(x=bound_input, IBP=False, method="CROWN", C=c, bound_upper=bound_upper)
    return lb, ub

# CROWN-IBP uses IBP bounds for all intermediate layers and CROWN bounds for the last one
def bound_crown_ibp(model, ptb, data, target, n_classes=10, bound_upper=False, reuse_input=False, loss_fusion=False):
    """Compute CROWN margin bounds using interval bounds at intermediate layers.

    CROWN-IBP combines inexpensive IBP intermediate bounds with a backward CROWN
    relaxation. Loss fusion instead bounds the supplied classifier-and-loss graph.

    Args:
        model (auto_LiRPA.BoundedModule): The neural network model for which bounds are to
            be computed.
        ptb (auto_LiRPA.PerturbationLpNorm): The perturbation object defining the
            perturbation set.
        data (Tensor): The input data tensor.
        target (Tensor): The target labels tensor.
        n_classes (int, optional): The number of classes for classification.
        bound_upper (bool, optional): Whether to compute the upper bound.
        reuse_input (bool, optional): Whether to reuse the input data from previous bounding
            operation.
        loss_fusion (bool, optional): Whether to use loss fusion.

    Returns:
        (tuple): Lower and optional upper bounds on the requested outputs or
            margins. The upper entry is None when upper bounds are not computed.
    """
    data = BoundedTensor(data, ptb=ptb)
    c = construct_c(data, target, n_classes) if not loss_fusion else None
    if reuse_input:
        bound_input = None
    elif loss_fusion:
        bound_input = (data, target)
    else:
        bound_input = (data,)
    lb, ub = model.compute_bounds(x=bound_input, IBP=False, method="CROWN-IBP", C=c, bound_upper=bound_upper)
    return lb, ub
