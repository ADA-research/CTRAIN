import torch
from auto_LiRPA import BoundedTensor
from CTRAIN.util import construct_c
from CTRAIN.bound.util import compute_ibp_bounds

def bound_ibp(model, ptb, data, target, n_classes=10, bound_upper=False, reuse_input=False, loss_fusion=False, relu_upper_retention=1.0):
    """
    Compute the lower and upper bounds of the model's output using the IBP method.

    Args:
        model (auto_LiRPA.BoundedModule): The neural network model for which bounds are to be computed.
        ptb (auto_LiRPA.PerturbationLpNorm): The perturbation object defining the perturbation set.
        data (Tensor): The input data tensor.
        target (Tensor, optional): The target labels tensor. Default is None.
        n_classes (int, optional): The number of classes for classification. Default is 10.
        bound_upper (bool, optional): Whether to compute the upper bound. Default is False.
        reuse_input (bool, optional): Whether to reuse the input data from previous bounding operation. Default is False.
        loss_fusion (bool, optional): Whether to use loss fusion. Default is False.
        relu_upper_retention (float): Unstable ReLU upper bound fraction for this training bound; default 1. Values below 1 are not sound certificates.

    Returns:
        (Tuple[Tensor, Tensor]): The lower and upper bounds of the model's output.
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
