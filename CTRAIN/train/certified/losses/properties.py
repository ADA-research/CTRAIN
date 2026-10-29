"""Certified property deficits and classification task losses."""
import torch
from torch.nn import functional as F


def advisory_cross_entropy(scores, target):
    """Preserve source ACAS Xu advisories through cross-entropy on negated scores.

    The lowest output score selects an advisory, so negating scores produces
    classification logits. Targets are integer source-model argmin advisories
    sampled before repair, with ties resolved by the first argmin. This empirical
    fidelity loss does not certify advisory preservation.

    Args:
        scores (torch.Tensor): Repaired model costs of shape (batch, actions).
        target (torch.Tensor): Source advisory indices of shape (batch,), with dtype
            torch.long.

    Returns:
        (torch.Tensor): Mean cross-entropy over the batch.
    """
    return F.cross_entropy(-scores, target)


def mnist_addition_cross_entropy(logits, target):
    """Average cross-entropy for independently predicted digits and their sum.

    Predictions concatenate heads of widths 10, 10 and 19; integer targets contain
    [a, b, a+b]. This task loss does not itself enforce arithmetic consistency.
    """
    if (logits.ndim != 2 or logits.shape[1] != 39
            or target.shape != (len(logits), 3) or target.dtype != torch.long):
        raise ValueError('Expected [batch,39] logits and [batch,3] long targets')
    heads = logits.split((10, 10, 19), dim=1)
    return torch.stack([F.cross_entropy(head, target[:, i])
                        for i, head in enumerate(heads)]).mean()


def property_loss(lower_bounds, margin=1e-3, *, loss='hinge', reduction='mean'):
    """Penalise deficits in property satisfaction margins.

    The hinge loss sums max(0, margin - lower_margin) over requirements and
    vanishes once each target margin is reached. Cross-entropy instead uses a
    zero-padded margin vector. Reduction averages over boxes unless set to none.
    """
    if not torch.isfinite(lower_bounds).all():
        raise FloatingPointError('Non-finite property bounds during training')
    if loss == 'cross_entropy':
        from CTRAIN.train.certified.losses.util import get_loss_from_bounds
        return get_loss_from_bounds(lower_bounds, torch.nn.CrossEntropyLoss(reduction='none'), reduction=reduction)
    if loss != 'hinge' or reduction not in ('mean', 'none'):
        raise ValueError('Invalid property loss or reduction')
    values = torch.relu(margin - lower_bounds).sum(dim=1)
    return values.mean() if reduction == 'mean' else values
