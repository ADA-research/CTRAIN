"""Translate lower classification margins into loss-function inputs."""
import torch

def get_loss_from_bounds(lb, criterion, *, reduction='mean'):
    """Compute a classification loss from lower bounds on true-class margins.

    Prepending a zero true-class logit to the negated margins produces the
    worst-case logit vector passed to criterion, with class zero as the target.

    Args:
        lb (torch.Tensor): A tensor containing the lower bounds with shape (batch_size,
            num_classes - 1).
        criterion (callable): A loss function that takes two arguments: the input tensor and
            the target tensor.
        reduction (str): 'mean' retains the scalar loss. 'none' requires a
            criterion returning one loss per example and preserves that vector.

    Returns:
        (torch.Tensor): Scalar mean loss, or per-example losses with reduction='none'.
    """
    lb_padded = torch.cat((torch.zeros(size=(lb.size(0),1), dtype=lb.dtype, device=lb.device), lb), dim=1)
    fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
    if reduction not in ('mean', 'none'):
        raise ValueError('reduction must be mean or none')
    certified_loss = criterion(-lb_padded, fake_labels)
    if reduction == 'mean':
        certified_loss = certified_loss.mean()
    return certified_loss
