"""IBP-R objective, equation (6) and optional masking in arXiv:2206.14772."""
import math

import torch
import torch.nn.functional as F
from auto_LiRPA.operators.relu import BoundRelu

from CTRAIN.attacks import pgd_attack
from CTRAIN.bound.ibp import bound_ibp
from CTRAIN.bound.util import get_pgd_ptb, validate_hybrid_options
from CTRAIN.util.util import preserve_model_state


def validate_ibp_r_options(rho, pgd_steps, pgd_restarts, pgd_alpha,
                          relu_upper_retention=1.0, pgd_eps_factor=1.0,
                          pgd_decay_factor=.1, pgd_decay_checkpoints=()):
    """Validate the IBP-R penalty weight, attack schedule and bound-retention options."""
    validate_hybrid_options(relu_upper_retention, pgd_eps_factor)
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be nonnegative and finite")
    for name, value in (("pgd_steps", pgd_steps), ("pgd_restarts", pgd_restarts)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(pgd_alpha) or pgd_alpha <= 0:
        raise ValueError("pgd_alpha must be positive and finite")
    if not math.isfinite(pgd_decay_factor) or pgd_decay_factor <= 0:
        raise ValueError("pgd_decay_factor must be positive and finite")
    if any(isinstance(step, bool) or not isinstance(step, int) or step < 0
           for step in pgd_decay_checkpoints):
        raise ValueError("PGD decay milestones must be nonnegative integers")


def relu_area(model, data):
    """Return each example's total unstable-ReLU relaxation area.

    For fresh IBP preactivation intervals [l, u], each unstable neuron contributes
    (-l) * u / 2. Contributions are summed across neurons and layers.
    """
    area = data.new_zeros(len(data))
    for node in model.modules():
        if isinstance(node, BoundRelu):
            lower, upper = node.inputs[0].lower, node.inputs[0].upper
            area = area + ((-lower).clamp(min=0) * upper.clamp(min=0)
                           / 2).reshape(len(data), -1).sum(1)
    return area


def get_ibp_r_loss(hardened_model, original_model, ptb, data, target,
                   kappa=1.0, rho=2e-4, regularisation_mask=False,
                   pgd_steps=8, pgd_alpha=.25, pgd_restarts=1,
                   pgd_eps_factor=1.0, relu_upper_retention=1.0,
                   data_min=None, data_max=None, eps=None,
                   pgd_early_stopping=False, pgd_decay_factor=.1,
                   pgd_decay_checkpoints=(), clean_output=None):
    """Combine adversarial cross-entropy with unstable-ReLU relaxation penalties.

    The clean-to-adversarial mixture is controlled by kappa. Fresh IBP intervals
    supply the regularisation areas, and optional masking selects correctly
    classified adversarial inputs. Returns the scalar loss and the predictions
    used for the adversarial component, or clean predictions at kappa=0.

    ``kappa`` is the *adversarial* weight (0 for clean, 1 after mixing).
    ``rho`` multiplies areas including their factor 1/2. Areas sum over
    neurons and layers, then average over the entire batch, even with masking.
    Both attack and IBP use the enlarged training box supplied in ``ptb``.
    ``pgd_eps_factor`` optionally scales only the attack; ``eps`` is the
    scheduled radius before clipping. L1 regularisation belongs to the trainer.
    Attacks run in eval mode; logits and bounds use the caller's module modes.
    Retention below 1 is an optional unsound surrogate, absent from the paper.

    ``kappa`` is the *adversarial* weight (0 for clean, 1 after mixing).
    ``rho`` multiplies areas including their factor 1/2. Areas sum over
    neurons and layers, then average over the entire batch, even with masking.
    Both attack and IBP use the enlarged training box supplied in ``ptb``.
    ``pgd_eps_factor`` optionally scales only the attack; ``eps`` is the
    scheduled radius before clipping. L1 regularisation belongs to the trainer.
    Attacks run in eval mode; logits and bounds use the caller's module modes.
    Retention below 1 is an optional unsound surrogate, absent from the paper.
    """
    validate_ibp_r_options(rho, pgd_steps, pgd_restarts, pgd_alpha,
                          relu_upper_retention, pgd_eps_factor,
                          pgd_decay_factor, pgd_decay_checkpoints)
    if not math.isfinite(kappa) or not 0 <= kappa <= 1:
        raise ValueError("kappa must be in [0, 1]")
    clean_loss = data.new_zeros(())
    if kappa < 1:
        if clean_output is None:
            clean_output = hardened_model(data)
        clean_loss = F.cross_entropy(clean_output, target)
    if kappa == 0:
        return clean_loss, clean_output
    attack_ptb = get_pgd_ptb(ptb, data, pgd_eps_factor, data_min, data_max, eps=eps)
    with preserve_model_state(original_model, hardened_model):
        original_model.eval()
        hardened_model.eval()
        adversarial = pgd_attack(
            hardened_model, data, target, attack_ptb.x_L, attack_ptb.x_U,
            restarts=pgd_restarts, step_size=pgd_alpha, n_steps=pgd_steps,
            early_stopping=pgd_early_stopping, device=data.device,
            decay_factor=pgd_decay_factor, decay_checkpoints=pgd_decay_checkpoints,
        )
    adv_output = hardened_model(adversarial)
    adv_loss = F.cross_entropy(adv_output, target)
    area = data.new_zeros(len(data))
    if rho:
        # Fresh auto_LiRPA intervals: never use attack or previous-batch caches.
        bound_ibp(hardened_model, ptb, data, None, bound_upper=True,
                  relu_upper_retention=relu_upper_retention)
        area = relu_area(hardened_model, data)
        if regularisation_mask:
            area = area * (adv_output.detach().argmax(1) == target)
    return (1 - kappa) * clean_loss + kappa * (adv_loss + rho * area.mean()), adv_output
