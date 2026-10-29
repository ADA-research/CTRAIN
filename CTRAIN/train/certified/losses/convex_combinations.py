"""Expressive losses combining adversarial searches and interval bounds."""
import torch
import math

from CTRAIN.attacks import pgd_attack
from CTRAIN.bound.util import get_pgd_ptb, validate_hybrid_options
from CTRAIN.train.certified.losses.ibp import get_ibp_loss
from CTRAIN.train.certified.losses.util import get_loss_from_bounds
from CTRAIN.util import construct_c
from CTRAIN.util.util import preserve_model_state


def validate_expressive_options(alpha, loss_type=None):
    """Validate the expressive-loss variant and its IBP mixture weight in [0, 1]."""
    if loss_type is not None and loss_type not in ('mtl_ibp', 'cc_ibp', 'exp_ibp'):
        raise ValueError('loss_type must be mtl_ibp, cc_ibp, or exp_ibp')
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be finite and in [0, 1]')

def get_expressive_ibp_loss(hardened_model, original_model, ptb, data, target, n_classes, criterion, alpha, return_bounds=False, return_stats=False, restarts=1, step_size=.2, n_steps=200, pgd_ptb=None, early_stopping=False, decay_checkpoints=(), decay_factor=.1, device='cuda', relu_upper_retention=1.0, pgd_eps_factor=1.0, data_min=None, data_max=None, eps=None, loss_type=None, *, loss_combination, teacher_features=None, distillation_weight=None):
    """Construct shared adversarial and IBP components for expressive losses.

    MTL-IBP combines their losses through a weighted sum, CC-IBP applies the
    criterion to their mixed classification margins, and Exp-IBP combines the
    batch-mean losses geometrically. The parameter alpha weights the IBP
    component. PGD runs in evaluation mode; loss evaluation uses the caller's
    mode. Returned IBP bounds concern the nominal input region and are sound
    only when relu_upper_retention is one. The published objectives use
    cross-entropy as the criterion.

    See De Palma et al. (2024), Expressive Losses for Verified Robustness via
    Convex Combinations: https://arxiv.org/abs/2305.13991.

    Args:
        hardened_model (auto_LiRPA.BoundedModule): The bounded model to be trained.
        original_model (torch.nn.Module): The original model.
        ptb (auto_LiRPA.PerturbationLpNorm): The perturbation applied to the input data.
        data (torch.Tensor): Input data.
        target (torch.Tensor): Target labels.
        n_classes (int): Number of classes.
        criterion (callable): Loss function.
        alpha (float): Weighting factor between robust loss and adversarial loss.
        return_bounds (bool, optional): If True, returns bounds.
        return_stats (bool, optional): If True, returns robustness and adversarial error
            statistics.
        restarts (int, optional): Number of restarts for PGD attack.
        step_size (float, optional): Step size for PGD attack.
        n_steps (int, optional): Number of steps for PGD attack.
        pgd_ptb (auto_LiRPA.PerturbationLpNorm, optional): PGD perturbation object.
        early_stopping (bool, optional): If True, stops early during PGD attack.
        decay_checkpoints (tuple, optional): Checkpoints for decay during PGD attack.
        decay_factor (float, optional): Decay factor for PGD attack.
        device (str, optional): Device to run the computations on.
        relu_upper_retention (float): Unstable ReLU upper bound fraction for this training
            bound; default 1. Values below 1 are not sound
            certificates.
        pgd_eps_factor (float): Positive PGD search-range factor; nominal IBP bounds are
            unchanged. Explicit pgd_ptb, when provided, takes
            precedence.
        data_min (torch.Tensor, optional): Lower input-domain limit in model coordinates.
        data_max (torch.Tensor, optional): Upper input-domain limit in model coordinates.
        eps (Tensor, optional): Perturbation radius in model input coordinates.
        loss_type (str, optional): Expressive-loss variant selected by the compatibility interface.
        loss_combination (callable): Callable combining bound-based and adversarial losses or margins.
        teacher_features (torch.nn.Module, optional): Frozen teacher feature extractor evaluated on clean inputs.
        distillation_weight (float, optional): Feature-distillation weight. None uses 5 divided by the feature
            width.

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
    if teacher_features is not None:
        feature_node = hardened_model[hardened_model.output_name[0]].inputs[0]
        adv_features = feature_node.forward_value
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

    if teacher_features is not None:
        from CTRAIN.train.certified.losses.cc_dist import feature_distillation_loss
        loss = loss + feature_distillation_loss(
            adv_features, feature_node.lower, feature_node.upper,
            teacher_features, alpha, distillation_weight)

    return_tuple = (loss,)

    if return_bounds:
        return_tuple = return_tuple + (lb, ub)
    if return_stats:
        robust_err = torch.sum(~(lb > 0).all(dim=1)).item() / data.size(0)
        adv_err = torch.sum(torch.argmax(adv_output, dim=1) != target).item() / data.size(0)
        return_tuple = return_tuple + (robust_err, adv_err)

    return return_tuple


def _combine_mtl(alpha, robust_loss, adv_loss, *_):
    """Combine IBP and adversarial losses through an alpha-weighted arithmetic mean."""
    return alpha * robust_loss + (1 - alpha) * adv_loss


def _combine_cc(alpha, robust_loss, adv_loss, lb, adv_output, data, target, n_classes, criterion):
    """Apply the criterion to alpha-weighted IBP and adversarial classification margins."""
    adv_lb = torch.bmm(construct_c(data, target, n_classes),
                       adv_output.unsqueeze(-1)).squeeze(-1)
    return get_loss_from_bounds(alpha * lb + (1 - alpha) * adv_lb, criterion)


def _combine_exp(alpha, robust_loss, adv_loss, *_):
    # Zero CE would otherwise produce undefined gradients in log space.
    """Combine positive batch-mean IBP and adversarial losses geometrically."""
    tiny = torch.finfo(robust_loss.dtype).tiny
    return (alpha * robust_loss.clamp_min(tiny).log()
            + (1 - alpha) * adv_loss.clamp_min(tiny).log()).exp()


def get_mtl_ibp_loss(hardened_model, original_model, ptb, data, target, n_classes, criterion, alpha, return_bounds=False, return_stats=False, restarts=1, step_size=.2, n_steps=200, pgd_ptb=None, early_stopping=False, decay_checkpoints=(), decay_factor=.1, device='cuda', relu_upper_retention=1.0, pgd_eps_factor=1.0, data_min=None, data_max=None, eps=None, loss_type='mtl_ibp'):
    """Compute the weighted sum of IBP and adversarial classification losses.

    Alpha weights the IBP component. Arguments and optional returns follow
    get_expressive_ibp_loss. The legacy loss_type argument also selects CC-IBP
    or Exp-IBP for callers using the earlier shared interface.
    """
    validate_expressive_options(alpha, loss_type)
    combination = {'mtl_ibp': _combine_mtl, 'cc_ibp': _combine_cc,
                   'exp_ibp': _combine_exp}[loss_type]
    options = locals().copy()
    options.pop('combination')
    return get_expressive_ibp_loss(**options, loss_combination=combination)


def get_cc_ibp_loss(*args, **kwargs):
    """Compute cross-entropy on mixed IBP and adversarial classification margins.

    Alpha weights the IBP margin. Arguments and optional returns follow
    get_expressive_ibp_loss.
    """
    return get_expressive_ibp_loss(*args, **kwargs, loss_type='cc_ibp', loss_combination=_combine_cc)


def get_exp_ibp_loss(*args, **kwargs):
    """Combine batch-mean IBP and adversarial losses through a geometric mixture.

    Alpha weights the logarithm of the IBP loss. Arguments and optional returns
    follow get_expressive_ibp_loss.
    """
    return get_expressive_ibp_loss(*args, **kwargs, loss_type='exp_ibp', loss_combination=_combine_exp)
