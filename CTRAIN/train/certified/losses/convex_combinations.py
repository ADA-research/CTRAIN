import torch
import numpy as np

from CTRAIN.attacks import pgd_attack
from CTRAIN.bound.util import get_pgd_ptb, validate_hybrid_options
from CTRAIN.train.certified.losses.ibp import get_ibp_loss

def get_mtl_ibp_loss(hardened_model, original_model, ptb, data, target, n_classes, criterion, alpha, return_bounds=False, return_stats=False, restarts=1, step_size=.2, n_steps=200, pgd_ptb=None, early_stopping=False, decay_checkpoints=(), decay_factor=.1, device='cuda', relu_upper_retention=1.0, pgd_eps_factor=1.0, data_min=None, data_max=None, eps=None):
    """
    Computes the MTL-IBP loss.

    Parameters:
        hardened_model (auto_LiRPA.BoundedModule): The bounded model to be trained.
        original_model (torch.nn.Module): The original model.
        ptb (autoLiRPA.PerturbationLpNorm): The perturbation applied to the input data.
        data (torch.Tensor): Input data.
        target (torch.Tensor): Target labels.
        n_classes (int): Number of classes.
        criterion (callable): Loss function.
        alpha (float): Weighting factor between robust loss and adversarial loss.
        return_bounds (bool, optional): If True, returns bounds. Default is False.
        return_stats (bool, optional): If True, returns robustness and adversarial error statistics. Default is False.
        restarts (int, optional): Number of restarts for PGD attack. Default is 1.
        step_size (float, optional): Step size for PGD attack. Default is 0.2.
        n_steps (int, optional): Number of steps for PGD attack. Default is 200.
        pgd_ptb (object, optional): PGD perturbation object. Default is None.
        early_stopping (bool, optional): If True, stops early during PGD attack. Default is False.
        decay_checkpoints (tuple, optional): Checkpoints for decay during PGD attack. Default is ().
        decay_factor (float, optional): Decay factor for PGD attack. Default is 0.1.
        device (str, optional): Device to run the computations on. Default is 'cuda'.

        relu_upper_retention (float): Unstable ReLU upper bound fraction for this training bound; default 1. Values below 1 are not sound certificates.
        pgd_eps_factor (float): Positive PGD search-range factor; nominal IBP bounds are unchanged. Explicit pgd_ptb, when provided, takes precedence.
        data_min, data_max (Tensor, optional): Pixel-domain limits for the scaled attack.
        eps (Tensor, optional): Scheduled radius before clipping; otherwise inferred from ptb bounds.

    Returns:
        (tuple): A tuple containing the loss, and optionally certified and adversarial error statistics.
    """
    validate_hybrid_options(relu_upper_retention, pgd_eps_factor)
    if pgd_ptb is None:
        pgd_ptb = get_pgd_ptb(ptb, data, pgd_eps_factor, data_min, data_max, eps=eps)
    hardened_model.eval()
    original_model.eval()

    x_adv = pgd_attack(
        model=hardened_model,
        data=data,
        target=target,
        x_L=pgd_ptb.x_L,
        x_U=pgd_ptb.x_U,
        restarts=restarts,
        step_size=step_size,
        n_steps=n_steps,
        early_stopping=early_stopping,
        device=device,
        decay_factor=decay_factor,
        decay_checkpoints=decay_checkpoints
    )

    hardened_model.train()
    original_model.train()

    adv_output = hardened_model(x_adv)
    adv_loss = criterion(adv_output, target).mean()

    robust_loss, lb, ub = get_ibp_loss(
        hardened_model=hardened_model,
        ptb=ptb,
        data=data,
        target=target,
        n_classes=n_classes,
        criterion=criterion,
        return_bounds=True,
        relu_upper_retention=relu_upper_retention
    )

    loss = alpha * robust_loss + (1 - alpha) * adv_loss

    return_tuple = (loss,)

    if return_bounds:
        assert False, "Return bounds is not implemented for MTL-IBP"
    elif return_stats:
        robust_err = torch.sum((lb < 0).any(dim=1)).item() / data.size(0)
        adv_err = torch.sum(torch.argmax(adv_output, dim=1) != target).item() / data.size(0)
        return_tuple = return_tuple + (robust_err, adv_err)

    return return_tuple
