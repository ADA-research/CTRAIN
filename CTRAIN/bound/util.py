"""Shared options for hybrid training bounds."""
import math
from types import MethodType

import torch
from auto_LiRPA import PerturbationLpNorm
from auto_LiRPA.operators.relu import BoundRelu


def validate_hybrid_options(relu_upper_retention=1.0, pgd_eps_factor=1.0):
    """Validate ReLU retention and PGD search-range factors before bound computation.

    Retention must lie in (0, 1] and the attack factor must be positive and finite.
    Invalid values raise ValueError.
    """
    if not math.isfinite(relu_upper_retention) or not 0 < relu_upper_retention <= 1:
        raise ValueError("relu_upper_retention must be in (0, 1]")
    if not math.isfinite(pgd_eps_factor) or pgd_eps_factor <= 0:
        raise ValueError("pgd_eps_factor must be positive and finite")


def get_pgd_ptb(ptb, data, factor=1.0, data_min=None, data_max=None, eps=None):
    """Construct an attack region independently of the nominal bounding region.

    The factor scales the search radius about data, with optional input-domain
    clipping. A factor of one returns the supplied perturbation unchanged.
    Pass the scheduled radius as eps when available: auto_LiRPA may replace a
    perturbation's eps with the half-width of an already clipped interval.
    """
    validate_hybrid_options(pgd_eps_factor=factor)
    if factor == 1:
        return ptb
    # auto_LiRPA replaces ptb.eps with half the clipped interval width.
    # Use the scheduled radius when supplied; otherwise infer it about data.
    if eps is None:
        eps = torch.maximum(data - ptb.x_L, ptb.x_U - data) if ptb.x_L is not None else ptb.eps
    eps = torch.as_tensor(eps, device=data.device, dtype=data.dtype)
    lower, upper = data - eps * factor, data + eps * factor
    lower = torch.clamp(lower, min=data_min, max=data_max) if data_min is not None or data_max is not None else lower
    upper = torch.clamp(upper, min=data_min, max=data_max) if data_min is not None or data_max is not None else upper
    return PerturbationLpNorm(norm=ptb.norm, eps=eps * factor, x_L=lower, x_U=upper)


def compute_ibp_bounds(model, relu_upper_retention=1.0, **kwargs):
    """Compute IBP bounds with optional training-only ReLU upper-bound retention.

    Ordinary IBP is used when retention is one. Smaller values multiply unstable
    ReLU upper bounds by the requested fraction, producing an unsound training
    surrogate. Since auto_LiRPA has no per-call option for this operation, its
    interval operators are temporarily adapted and restored even on failure.
    Arguments in kwargs are forwarded to model.compute_bounds.
    """
    validate_hybrid_options(relu_upper_retention)
    if relu_upper_retention == 1:
        return model.compute_bounds(**kwargs)
    originals = []
    try:
        for node in model.modules():
            if isinstance(node, BoundRelu):
                original = node.interval_propagate
                originals.append((node, original, "interval_propagate" in node.__dict__))

                def interval(self, *v, _original=original):
                    lower, upper = _original(*v)
                    unstable = (v[0][0] < 0) & (v[0][1] > 0)
                    return lower, torch.where(unstable, upper * relu_upper_retention, upper)

                node.interval_propagate = MethodType(interval, node)
        return model.compute_bounds(**kwargs)
    finally:
        for node, original, overridden in originals:
            if overridden:
                node.interval_propagate = original
            else:
                del node.interval_propagate
