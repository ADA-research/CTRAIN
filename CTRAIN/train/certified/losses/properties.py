"""Certified property deficits and classification task losses."""
import torch
from torch.nn import functional as F


def advisory_cross_entropy(scores, target):
    """Preserve source advisories for models whose lowest score selects an action.

    Negate scores to obtain classification logits: a lower cost then has higher
    probability. Targets are fixed source-model ``argmin`` labels sampled before
    repair, not source score vectors or labels regenerated during training.
    This empirical fidelity loss does not certify advisory preservation.

    Args:
        scores (torch.Tensor): Repaired model costs of shape ``(batch, actions)``.
        target (torch.Tensor): Source advisory indices of shape ``(batch,)``,
            with dtype ``torch.long``. Source ties use PyTorch's first argmin.

    Returns:
        torch.Tensor: Mean cross-entropy over the batch.
    """
    return F.cross_entropy(-scores, target)


def mnist_addition_cross_entropy(logits, target):
    """Mean CE for independent digit/digit/sum heads and integer [a,b,a+b] labels."""
    if (logits.ndim != 2 or logits.shape[1] != 39
            or target.shape != (len(logits), 3) or target.dtype != torch.long):
        raise ValueError('Expected [batch,39] logits and [batch,3] long targets')
    heads = logits.split((10, 10, 19), dim=1)
    return torch.stack([F.cross_entropy(head, target[:, i])
                        for i, head in enumerate(heads)]).mean()


def property_loss(lower_bounds, margin=1e-3, *, loss='hinge', reduction='mean'):
    """Property margin loss, averaged over regions unless reduction is none.

    Hinge sums deficits over obligations; cross-entropy uses the existing
    padded-margin loss and ignores the hinge slack parameter.
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
