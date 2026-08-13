import torch
from CTRAIN.bound import bound_sabr
from CTRAIN.train.certified.losses import get_loss_from_bounds

def get_sabr_loss(hardened_model, original_model, data, target, eps, subselection_ratio, criterion, device='cuda', 
                      n_classes=10, x_L=None, x_U=None, data_min=None, data_max=None, pgd_steps=8, pgd_step_size=.5, pgd_restarts=1, pgd_early_stopping=True, pgd_decay_factor=.1, pgd_decay_checkpoints=(4,7), pgd_ptb=None, return_stats=False, return_bounds=False, **kwargs):

    """
    Compute the SABR loss for a given model and data.
    
    Parameters:
        hardened_model (auto_LiRPA.BoundedModule): The bounded model to be trained.
        original_model (torch.nn.Module): The original model.
        data (torch.Tensor): Input data.
        target (torch.Tensor): Target labels.
        eps (float): Perturbation budget for adversarial attacks.
        subselection_ratio (float): Factor to shrink epsilon by.
        criterion (callable): Loss function to use.
        device (str, optional): Device to run the computation on. Default is 'cuda'.
        n_classes (int, optional): Number of classes. Default is 10.
        x_L (torch.Tensor, optional): Lower bound for input data. Default is None.
        x_U (torch.Tensor, optional): Upper bound for input data. Default is None.
        data_min (torch.Tensor, optional): Minimum value for input data. Default is None.
        data_max (torch.Tensor, optional): Maximum value for input data. Default is None.
        pgd_steps (int, optional): Number of PGD steps. Default is 8.
        pgd_step_size (float, optional): Step size for PGD. Default is 0.5.
        pgd_restarts (int, optional): Number of PGD restarts. Default is 1.
        pgd_early_stopping (bool, optional): Whether to use early stopping in PGD. Default is True.
        pgd_decay_factor (float, optional): Decay factor for PGD step size. Default is 0.1.
        pgd_decay_checkpoints (tuple, optional): Checkpoints for PGD step size decay. Default is (4, 7).
        pgd_ptb (PerturbationLpNorm, optional): Perturbation bounds used for the
            PGD center search. The certified SABR subregion remains constrained
            to the nominal bounds defined by ``eps`` or ``x_L``/``x_U``.
        return_stats (bool, optional): Whether to return robustness statistics. Default is False.
        return_bounds (bool, optional): Whether to return bounds. Default is False.
        **kwargs: Additional arguments.
    
    Returns:
        (tuple): A tuple containing the loss, and optionally robustness statistics (robust_err, adv_err).
    """
    lb, ub, adv_output = bound_sabr(
        hardened_model=hardened_model,
        original_model=original_model,
        data=data,
        data_min=data_min,
        data_max=data_max,
        target=target,
        eps=eps,
        subselection_ratio=subselection_ratio,
        device=device,
        n_classes=n_classes,
        x_L=x_L,
        x_U=x_U,
        n_steps=pgd_steps,
        step_size=pgd_step_size,
        restarts=pgd_restarts,
        early_stopping=pgd_early_stopping,
        decay_checkpoints=pgd_decay_checkpoints, 
        decay_factor=pgd_decay_factor,
        pgd_ptb=pgd_ptb,
        return_adv_output=True
    )
    loss = get_loss_from_bounds(lb, criterion=criterion)
    
    return_tuple = (loss,)
    
    if return_bounds:
        assert False, "Return bounds is not implemented for MTL-IBP"
    elif return_stats:
        robust_err = torch.sum((lb < 0).any(dim=1)).item() / data.size(0)
        adv_err = torch.sum(torch.argmax(adv_output, dim=1) != target).item() / data.size(0)
        return_tuple = return_tuple + (robust_err, adv_err)
    
    return return_tuple
