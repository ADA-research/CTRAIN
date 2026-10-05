import copy
import torch
from auto_LiRPA import BoundedTensor

from CTRAIN.util import construct_c
from CTRAIN.util.util import preserve_model_state


def bound_alpha_crown(model, ptb, data, target, n_classes=10,
                      bound_upper=False, reuse_input=False, alpha_iterations=20):
    """Optimize CROWN's ReLU slopes; restore caller modes, flags and options.

    Parameter gradients are left untouched. Optimization requires gradients even
    when called inside ``torch.no_grad()``. These are sound bounds, not shrinkage.
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
