"""MTL-IBP compatibility entry point for the common expressive IBP trainer."""
import torch
from CTRAIN.train.certified.expressive_ibp import expressive_ibp_train_model
from CTRAIN.train.certified.losses import get_mtl_ibp_loss, get_cc_ibp_loss, get_exp_ibp_loss
from CTRAIN.train.certified.losses.convex_combinations import validate_expressive_options


def mtl_ibp_train_model(
    original_model,
    hardened_model,
    train_loader,
    val_loader=None,
    start_epoch=0,
    end_epoch=None,
    num_epochs=None,
    eps=0.3,
    eps_std=0.3,
    eps_schedule=(0, 20, 50),
    eps_schedule_unit="epoch",
    eps_scheduler_args=dict(),
    optimizer=None,
    lr_scheduler=None,
    lr_decay_schedule_unit="epoch",
    n_classes=10,
    gradient_clip=None,
    shi_regularisation_weight=0.5,
    shi_reg_decay=True,
    l1_regularisation_weight=0.00001,
    alpha=0.5,
    pgd_restarts=1,
    pgd_step_size=10,
    pgd_n_steps=1,
    pgd_eps_factor=1,
    pgd_decay_factor=0.1,
    pgd_decay_checkpoints=(),
    pgd_early_stopping=False,
    results_path="./results",
    checkpoint_save_interval=10,
    device="cuda",
    relu_upper_retention=1.0,
    population_bn=False,
    loss_type='mtl_ibp', validation_eps=None, selection_state=None, *, history=None):
    """Train MTL-IBP; existing arguments and legacy loss_type selection are preserved."""
    validate_expressive_options(alpha, loss_type)
    options = locals().copy()
    options.pop('loss_type')
    loss_func = {'mtl_ibp': get_mtl_ibp_loss, 'cc_ibp': get_cc_ibp_loss,
                 'exp_ibp': get_exp_ibp_loss}[loss_type]
    return expressive_ibp_train_model(**options, loss_func=loss_func, method=loss_type)
