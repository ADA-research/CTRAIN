"""Feature-space CC-Dist objective, arXiv:2602.02626, equations (8)–(9)."""
import math

import torch
from torch import nn
from auto_LiRPA.operators.normalization import BoundBatchNormalization

from CTRAIN.train.certified.losses.convex_combinations import get_cc_ibp_loss, validate_expressive_options
from CTRAIN.util.util import preserve_model_state


def validate_cc_dist_bn_options(model, bound_opts):
    """Validate that CC-Dist bounds reuse the adversarial forward pass's BatchNorm statistics."""
    bound_opts = {} if bound_opts is None else bound_opts
    has_bn = any(isinstance(m, (nn.modules.batchnorm._BatchNorm, BoundBatchNormalization))
                 for m in model.modules())
    if has_bn and (bound_opts.get('forward_before_compute_bounds', False)
                   or bound_opts.get('bn', {}).get('mode', 'forward') != 'forward'):
        raise ValueError('CC-Dist BatchNorm requires forward_before_compute_bounds=False '
                         'and bn.mode=forward')


def feature_distillation_loss(adversarial, lower, upper, teacher, alpha, weight=None):
    """Align adversarial and bounded student features with clean teacher features.

    The loss averages the sum of the worst squared endpoint distances after
    mixing adversarial features and interval endpoints with weight alpha.
    A missing weight uses 5 divided by the feature width. Teacher features are
    detached, so the loss updates only the student.
    """
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be finite and in [0, 1]')
    if weight is not None and (not math.isfinite(weight) or weight < 0):
        raise ValueError('distillation_weight must be finite and nonnegative')
    if not (adversarial.shape == lower.shape == upper.shape == teacher.shape):
        raise ValueError('Teacher and student feature shapes must match')
    width = adversarial[0].numel()
    beta = 5 / width if weight is None else weight
    mixed_lower = (1 - alpha) * adversarial + alpha * lower
    mixed_upper = (1 - alpha) * adversarial + alpha * upper
    distances = torch.maximum((mixed_lower - teacher.detach()).square(),
                              (mixed_upper - teacher.detach()).square())
    return beta * distances.flatten(1).sum(1).mean()


def get_cc_dist_loss(*args, teacher_features, distillation_weight=None, **kwargs):
    """Extend CC-IBP with feature distillation from an adversarially trained teacher.

    The frozen teacher is evaluated on clean inputs in evaluation mode. Student
    features and their bounds come from the same CC-IBP graph and adversarial
    BatchNorm pass. Setting distillation_weight=0 gives the CC-IBP control.
    Other arguments and optional returns follow get_expressive_ibp_loss.
    """
    if distillation_weight is not None and (
            not math.isfinite(distillation_weight) or distillation_weight < 0):
        raise ValueError('distillation_weight must be finite and nonnegative')
    validate_expressive_options(kwargs['alpha'] if 'alpha' in kwargs else args[7])
    model = kwargs['hardened_model'] if 'hardened_model' in kwargs else args[0]
    if model.training:
        validate_cc_dist_bn_options(model, model.bound_opts)
    if distillation_weight == 0:
        return get_cc_ibp_loss(*args, **kwargs)
    data = kwargs['data'] if 'data' in kwargs else args[3]
    with preserve_model_state(teacher_features), torch.no_grad():
        teacher_features.eval()
        features = teacher_features(data).detach()
    return get_cc_ibp_loss(*args, **kwargs, teacher_features=features,
                           distillation_weight=distillation_weight)
