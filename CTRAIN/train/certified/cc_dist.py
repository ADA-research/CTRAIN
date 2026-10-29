"""CC-Dist uses the shared expressive schedules, regularization and checkpoints."""
from functools import partial

from CTRAIN.train.certified.expressive_ibp import expressive_ibp_train_model
from CTRAIN.train.certified.losses.cc_dist import get_cc_dist_loss, validate_cc_dist_bn_options


def cc_dist_train_model(*args, teacher_features=None, distillation_weight=None, **kwargs):
    """Train CC-Dist; arguments otherwise match expressive_ibp_train_model."""
    model = kwargs['hardened_model'] if 'hardened_model' in kwargs else args[1]
    validate_cc_dist_bn_options(model, model.bound_opts)
    if 'loss_func' not in kwargs:
        if teacher_features is None:
            raise ValueError('CC-Dist requires teacher_features')
        kwargs['loss_func'] = partial(get_cc_dist_loss, teacher_features=teacher_features,
                                      distillation_weight=distillation_weight)
    kwargs['method'] = 'cc_dist'
    kwargs['clean_forward'] = False
    return expressive_ibp_train_model(*args, **kwargs)
