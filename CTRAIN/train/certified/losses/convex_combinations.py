import torch
import math

from CTRAIN.attacks import pgd_attack
from CTRAIN.bound.util import get_pgd_ptb, validate_hybrid_options
from CTRAIN.train.certified.losses.ibp import get_ibp_loss
from CTRAIN.train.certified.losses.util import get_loss_from_bounds
from CTRAIN.util import construct_c
from CTRAIN.util.util import preserve_model_state


def validate_expressive_options(alpha, loss_type=None):
    if loss_type is not None and loss_type not in ('mtl_ibp', 'cc_ibp', 'exp_ibp'):
        raise ValueError('loss_type must be mtl_ibp, cc_ibp, or exp_ibp')
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be finite and in [0, 1]')

def get_expressive_ibp_loss(hardened_model, original_model, ptb, data, target, n_classes, criterion, alpha, return_bounds=False, return_stats=False, restarts=1, step_size=.2, n_steps=200, pgd_ptb=None, early_stopping=False, decay_checkpoints=(), decay_factor=.1, device='cuda', relu_upper_retention=1.0, pgd_eps_factor=1.0, data_min=None, data_max=None, eps=None, loss_type=None, *, loss_combination):
    """
    Compute shared PGD/IBP components and apply the supplied loss combination.

    CC-IBP mixes true-class margins before cross-entropy; Exp-IBP takes the
    geometric mixture of batch-mean losses, as in the authors' train.py:
    https://github.com/alessandrodepalma/expressive-losses/blob/main/train.py
    Returned bounds are the IBP bounds, not the mixed margins or certificates
    of the adversarial component. ``criterion`` must be cross-entropy for the
    published objectives.

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
    validate_expressive_options(alpha, loss_type)
    if pgd_ptb is None:
        pgd_ptb = get_pgd_ptb(ptb, data, pgd_eps_factor, data_min, data_max, eps=eps)
    with preserve_model_state(hardened_model, original_model):
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

    if alpha == 0:
        loss = adv_loss
    elif alpha == 1:
        loss = robust_loss
    else:
        loss = loss_combination(alpha, robust_loss, adv_loss, lb, adv_output,
                                data, target, n_classes, criterion)

    return_tuple = (loss,)

    if return_bounds:
        return_tuple = return_tuple + (lb, ub)
    if return_stats:
        robust_err = torch.sum((lb < 0).any(dim=1)).item() / data.size(0)
        adv_err = torch.sum(torch.argmax(adv_output, dim=1) != target).item() / data.size(0)
        return_tuple = return_tuple + (robust_err, adv_err)

    return return_tuple


def _combine_mtl(alpha, robust_loss, adv_loss, *_):
    return alpha * robust_loss + (1 - alpha) * adv_loss


def _combine_cc(alpha, robust_loss, adv_loss, lb, adv_output, data, target, n_classes, criterion):
    adv_lb = torch.bmm(construct_c(data, target, n_classes),
                       adv_output.unsqueeze(-1)).squeeze(-1)
    return get_loss_from_bounds(alpha * lb + (1 - alpha) * adv_lb, criterion)


def _combine_exp(alpha, robust_loss, adv_loss, *_):
    # Zero CE would otherwise produce undefined gradients in log space.
    tiny = torch.finfo(robust_loss.dtype).tiny
    return (alpha * robust_loss.clamp_min(tiny).log()
            + (1 - alpha) * adv_loss.clamp_min(tiny).log()).exp()


def get_mtl_ibp_loss(hardened_model, original_model, ptb, data, target, n_classes, criterion, alpha, return_bounds=False, return_stats=False, restarts=1, step_size=.2, n_steps=200, pgd_ptb=None, early_stopping=False, decay_checkpoints=(), decay_factor=.1, device='cuda', relu_upper_retention=1.0, pgd_eps_factor=1.0, data_min=None, data_max=None, eps=None, loss_type='mtl_ibp'):
    """MTL-IBP loss; legacy loss_type selection remains supported."""
    validate_expressive_options(alpha, loss_type)
    combination = {'mtl_ibp': _combine_mtl, 'cc_ibp': _combine_cc,
                   'exp_ibp': _combine_exp}[loss_type]
    options = locals().copy()
    options.pop('combination')
    return get_expressive_ibp_loss(**options, loss_combination=combination)


def get_cc_ibp_loss(*args, **kwargs):
    """CC-IBP margin mixture; shared arguments match get_expressive_ibp_loss."""
    return get_expressive_ibp_loss(*args, **kwargs, loss_type='cc_ibp', loss_combination=_combine_cc)


def get_exp_ibp_loss(*args, **kwargs):
    """Exp-IBP batch-loss mixture; shared arguments match get_expressive_ibp_loss."""
    return get_expressive_ibp_loss(*args, **kwargs, loss_type='exp_ibp', loss_combination=_combine_exp)
