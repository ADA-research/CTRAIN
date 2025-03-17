import torch
from auto_LiRPA import BoundedTensor

from CTRAIN.util import construct_c

# Uses CROWN bounds throughout all intermediate layers and the final layer. 
def bound_crown(model, ptb, data, target, n_classes=10, bound_upper=False, reuse_input=False):
    """
    Compute the lower and upper bounds of the model's output using the CROWN method.

    Parameters:
        model (auto_LiRPA.BoundedModule): The neural network model for which bounds are to be computed.
        ptb (auto_LiRPA.PerturbationLpNorm): The perturbation object defining the perturbation set.
        data (Tensor): The input data tensor.
        target (Tensor): The target labels tensor.
        n_classes (int, optional): The number of classes for classification. Default is 10.
        bound_upper (bool, optional): Whether to compute the upper bound. Default is False.
        reuse_input (bool, optional): Whether to reuse the input data from previous bounding operation. Default is False.

    Returns:
        (Tuple[Tensor, Tensor]): The lower and upper bounds of the model's output.
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
    """
    Compute the lower and upper bounds of the model's output using the CROWN-IBP method.

    Parameters:
        model (auto_LiRPA.BoundedModule): The neural network model for which bounds are to be computed.
        ptb (auto_LiRPA.PerturbationLpNorm): The perturbation object defining the perturbation set.
        data (Tensor): The input data tensor.
        target (Tensor): The target labels tensor.
        n_classes (int, optional): The number of classes for classification. Default is 10.
        bound_upper (bool, optional): Whether to compute the upper bound. Default is False.
        reuse_input (bool, optional): Whether to reuse the input data from previous bounding operation. Default is False.
        loss_fusion (bool, optional): Whether to use loss fusion. Default is False.

    Returns:
        (Tuple[Tensor, Tensor]): The lower and upper bounds of the model's output.
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