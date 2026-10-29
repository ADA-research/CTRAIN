"""MTL-IBP wrapper using the common expressive IBP training and HPO infrastructure."""
import torch

from CTRAIN.model_wrappers.certified.expressive_ibp_model_wrapper import ExpressiveIBPModelWrapper
from CTRAIN.train.certified.losses import get_mtl_ibp_loss


class MTLIBPModelWrapper(ExpressiveIBPModelWrapper):
    """Train MTL-IBP from De Palma et al., Expressive Losses (ICLR 2024).

    ``mtl_ibp_alpha`` weights IBP loss in an arithmetic loss mixture; it must be in
    [0, 1]. All other options have the same meaning as in ExpressiveIBPModelWrapper.
    Bounds are computed by auto_LiRPA. Training surrogates are not certificates.
    """
    _loss_type = 'mtl_ibp'
    _alpha_name = 'mtl_ibp_alpha'
    _loss_func = staticmethod(get_mtl_ibp_loss)

    def __init__(self, model, input_shape, eps, num_epochs, train_eps_factor=1,
                 optimizer_func=torch.optim.Adam, lr=0.0005, warm_up_epochs=1, ramp_up_epochs=70,
                 lr_scheduler_func=torch.optim.lr_scheduler.MultiStepLR,
                 lr_decay_kwargs=dict(milestones=(80, 90)), gradient_clip=10, l1_reg_weight=0.000001,
                 shi_reg_weight=.5, shi_reg_decay=True, pgd_steps=1, pgd_alpha=10, pgd_restarts=1,
                 pgd_early_stopping=False, pgd_alpha_decay_factor=.1, pgd_decay_milestones=(),
                 pgd_eps_factor=1, mtl_ibp_alpha=.5, checkpoint_save_path=None, checkpoint_save_interval=10,
                 bound_opts=dict(conv_mode='patches', relu='adaptive'), device=torch.device('cuda'),
                 population_bn=False, relu_upper_retention=1.0):
        super().__init__(
            model, input_shape, eps, num_epochs, train_eps_factor, optimizer_func, lr,
            warm_up_epochs, ramp_up_epochs, lr_scheduler_func, lr_decay_kwargs, gradient_clip,
            l1_reg_weight, shi_reg_weight, shi_reg_decay, pgd_steps, pgd_alpha, pgd_restarts,
            pgd_early_stopping, pgd_alpha_decay_factor, pgd_decay_milestones, pgd_eps_factor,
            mtl_ibp_alpha, checkpoint_save_path, checkpoint_save_interval, bound_opts, device,
            population_bn, relu_upper_retention)
