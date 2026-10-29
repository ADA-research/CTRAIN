"""CC-IBP entry point for the common expressive IBP trainer."""
from CTRAIN.train.certified.expressive_ibp import expressive_ibp_train_model
from CTRAIN.train.certified.losses import get_cc_ibp_loss


def cc_ibp_train_model(*args, **kwargs):
    """Train CC-IBP using the shared expressive IBP training procedure.

    Arguments follow expressive_ibp_train_model; this entry point selects the
    method-specific loss while reusing its schedules and training loop.
    """
    return expressive_ibp_train_model(*args, **kwargs, loss_func=get_cc_ibp_loss, method='cc_ibp')
