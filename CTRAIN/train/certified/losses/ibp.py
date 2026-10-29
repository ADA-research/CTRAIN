"""Classification losses derived from interval lower bounds on true-class margins."""
import torch
from CTRAIN.bound import bound_ibp
from CTRAIN.train.certified.losses import get_loss_from_bounds

def get_ibp_loss(hardened_model, ptb, data, target, n_classes, criterion, return_bounds=False, return_stats=False, relu_upper_retention=1.0):
    """Compute the certified classification loss from IBP margin bounds.

    The lower bound on each true-class margin defines a worst-case logit vector.
    Applying the criterion to this vector encourages correct classification
    throughout the input region. ReLU retention below one replaces this bound
    with an unsound training surrogate.

    Args:
        hardened_model (auto_LiRPA.BoundedModule): The bounded model to be trained.
        ptb (auto_LiRPA.PerturbationLpNorm): The perturbation applied to the input data.
        data (torch.Tensor): Input data.
        target (torch.Tensor): Target labels.
        n_classes (int): Number of classes.
        criterion (callable): Loss function to be used.
        return_bounds (bool, optional): If True, return the lower and upper bounds.
        return_stats (bool, optional): If True, return additional statistics.
        relu_upper_retention (float): Unstable ReLU upper bound fraction for this training
            bound; default 1. Values below 1 are not sound
            certificates.

    Returns:
        (tuple): A tuple containing the certified loss. If `return_bounds` is True, the tuple also contains the lower and upper bounds. 
               If `return_stats` is True, the tuple also contains the robust error.
    """
    lb, ub = bound_ibp(
        model=hardened_model,
        ptb=ptb,
        data=data,
        target=target,
        n_classes=n_classes,
        relu_upper_retention=relu_upper_retention,
    )
    certified_loss = get_loss_from_bounds(lb, criterion)
    
    return_tuple = (certified_loss,)
    
    if return_bounds:
        return_tuple = return_tuple + (lb, ub)
    if return_stats:
        robust_err = torch.sum(~(lb > 0).all(dim=1)).item() / data.size(0)
        return_tuple = return_tuple + (robust_err,)
    
    return return_tuple
