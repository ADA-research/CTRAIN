"""Compute interval bounds for certified losses and incomplete verification."""
import torch
from auto_LiRPA import BoundedTensor
from CTRAIN.util import construct_c
from CTRAIN.bound.util import compute_ibp_bounds

def bound_ibp(model, ptb, data, target, n_classes=10, bound_upper=False, reuse_input=False, loss_fusion=False, relu_upper_retention=1.0):
    """Compute interval bounds on outputs or true-class classification margins.

    IBP propagates an input interval through the network using auto_LiRPA.
    When target labels are supplied, the bounded quantities are the true-class
    logits minus each competing logit. With target=None, raw outputs are bounded.
    ReLU retention below one is intended only as an unsound training surrogate.

    Args:
        model (auto_LiRPA.BoundedModule): The neural network model for which bounds are to
            be computed.
        ptb (auto_LiRPA.PerturbationLpNorm): The perturbation object defining the
            perturbation set.
        data (Tensor): The input data tensor.
        target (Tensor, optional): The target labels tensor.
        n_classes (int, optional): The number of classes for classification.
        bound_upper (bool, optional): Whether to compute the upper bound.
        reuse_input (bool, optional): Whether to reuse the input data from previous bounding
            operation.
        loss_fusion (bool, optional): Whether to use loss fusion.
        relu_upper_retention (float): Unstable ReLU upper bound fraction for this training
            bound; default 1. Values below 1 are not sound
            certificates.

    Returns:
        (tuple): Lower and optional upper bounds on the requested outputs or
            margins. The upper entry is None when upper bounds are not computed.
    """
    data = BoundedTensor(data, ptb=ptb)
    if target is not None and not loss_fusion:
        c = construct_c(data, target, n_classes)
    else:
        c = None
    if reuse_input:
        bound_input = None
    elif loss_fusion:
        bound_input = (data, target)
    else:
        bound_input = (data,)
    lb, ub = compute_ibp_bounds(
        model, relu_upper_retention, x=bound_input, IBP=True, method="IBP",
        C=c, bound_upper=bound_upper)
    return lb, ub
