import torch
from auto_LiRPA import BoundedTensor
from CTRAIN.util import construct_c

def bound_ibp(model, ptb, data, target, n_classes=10, bound_upper=False, reuse_input=False, loss_fusion=False):
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
    lb, ub = model.compute_bounds(x=bound_input, IBP=True, method="IBP", C=c, bound_upper=bound_upper)
    return lb, ub