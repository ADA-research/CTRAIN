"""Feature-space CC-Dist objective, arXiv:2602.02626, equations (8)–(9)."""
import math

import torch
from torch import nn
from auto_LiRPA.operators.normalization import BoundBatchNormalization

from CTRAIN.train.certified.losses.convex_combinations import get_cc_ibp_loss, validate_expressive_options
from CTRAIN.util.util import preserve_model_state


def validate_cc_dist_bn_options(model, bound_opts):
    """Bounds must reuse the adversarial forward's BatchNorm statistics."""
    bound_opts = {} if bound_opts is None else bound_opts
    has_bn = any(isinstance(m, (nn.modules.batchnorm._BatchNorm, BoundBatchNormalization))
                 for m in model.modules())
    if has_bn and (bound_opts.get('forward_before_compute_bounds', False)
                   or bound_opts.get('bn', {}).get('mode', 'forward') != 'forward'):
        raise ValueError('CC-Dist BatchNorm requires forward_before_compute_bounds=False '
                         'and bn.mode=forward')


def feature_distillation_loss(adversarial, lower, upper, teacher, alpha, weight=None):
    """Batch mean of the sum of worst endpoint distances, with beta=5/w."""
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
    """CC-IBP plus distillation from frozen, clean, evaluation-mode features.

    The attack and the bounds use the existing CC-IBP implementation. Features
    and their intervals come from its same LiRPA graph and adversarial BN pass.
    ``teacher_features`` is the teacher without its final affine classifier.
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
