"""Exp-IBP entry point for the common expressive IBP trainer."""
from CTRAIN.train.certified.expressive_ibp import expressive_ibp_train_model
from CTRAIN.train.certified.losses import get_exp_ibp_loss


def exp_ibp_train_model(*args, **kwargs):
    """Train Exp-IBP; shared arguments match expressive_ibp_train_model."""
    return expressive_ibp_train_model(*args, **kwargs, loss_func=get_exp_ibp_loss, method='exp_ibp')
