"""Gaussian loss smoothing with existing auto_LiRPA certified losses."""
import copy
import math
import torch
from smac.utils.configspace import get_config_hash
from CTRAIN.model_wrappers.certified.model_wrapper import CTRAINWrapper
from CTRAIN.train.certified.gls import GLSOptimizer, gls_train_model
from CTRAIN.util import seed_ctrain


class GLSModelWrapper(CTRAINWrapper):
    """RGS or PGPE training (arXiv:2403.07095).

    CROWN corresponds to adaptive DeepPoly; CROWN-IBP uses interval intermediate
    bounds. PGPE paper experiments start from adversarially pretrained weights:
    load those weights into ``model`` and pass ``shi_init=False``. This adapter
    runs samples sequentially, without a distributed population dependency.
    """
    def __init__(self, model, input_shape, eps, num_epochs, train_eps_factor=1,
                 optimizer_func=torch.optim.Adam, lr=5e-5, warm_up_epochs=1,
                 ramp_up_epochs=80, lr_scheduler_func=torch.optim.lr_scheduler.MultiStepLR,
                 lr_decay_kwargs=None, checkpoint_save_path=None,
                 checkpoint_save_interval=10, bound_opts=None, device='cuda',
                 population_bn=False, estimator='rgs', samples=2, sigma=1e-3,
                 sigma_lr=1e-3, sigma_decay_milestones=(), sigma_decay_factor=.4,
                 bound_method='CROWN', shi_init=True, shi_reg_weight=0.,
                 shi_reg_decay=True, l1_reg_weight=0., gradient_clip=None):
        if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(sigma_lr) or sigma_lr < 0:
            raise ValueError('invalid Gaussian scale or scale learning rate')
        if gradient_clip is not None and (not math.isfinite(gradient_clip) or gradient_clip <= 0):
            raise ValueError('gradient_clip must be positive finite')
        if type(num_epochs) is not int or num_epochs < 1:
            raise ValueError('num_epochs must be a positive integer')
        if (type(warm_up_epochs) is not int or type(ramp_up_epochs) is not int
                or warm_up_epochs < 0 or ramp_up_epochs < 0
                or warm_up_epochs + ramp_up_epochs > num_epochs):
            raise ValueError('invalid warm-up/ramp-up schedule')
        if bound_method not in ('IBP', 'CROWN', 'CROWN-IBP'):
            raise ValueError('unsupported bound_method')
        if any(type(m) is not int or m < 1 for m in sigma_decay_milestones):
            raise ValueError('sigma decay milestones must be positive integer epochs')
        if estimator == 'pgpe' and sigma_decay_milestones:
            raise ValueError('sigma_decay_milestones is for fixed-scale RGS; PGPE learns its scale')
        if checkpoint_save_interval < 1:
            raise ValueError('checkpoint_save_interval must be positive')
        if not math.isfinite(shi_reg_weight) or shi_reg_weight < 0 or not math.isfinite(l1_reg_weight) or l1_reg_weight < 0:
            raise ValueError('regularization weights must be nonnegative finite')
        if not 0 < sigma_decay_factor <= 1:
            raise ValueError('sigma_decay_factor must be in (0, 1]')
        if estimator not in ('rgs', 'pgpe') or type(samples) is not int or samples < 1 or (estimator == 'pgpe' and samples % 2):
            raise ValueError('invalid GLS estimator/sample count')
        self.num_epochs = num_epochs
        self.options = dict(warm_up_epochs=warm_up_epochs, ramp_up_epochs=ramp_up_epochs,
            sigma_decay_milestones=tuple(sigma_decay_milestones), sigma_decay_factor=sigma_decay_factor,
            bound_method=bound_method, shi_init=shi_init, shi_reg_weight=shi_reg_weight,
            shi_reg_decay=shi_reg_decay, l1_reg_weight=l1_reg_weight)
        decay = dict(milestones=(120, 140), gamma=.2) if lr_decay_kwargs is None else lr_decay_kwargs
        super().__init__(model, eps, input_shape, train_eps_factor, lr, optimizer_func,
                         lr_scheduler_func, decay,
                         dict(conv_mode='patches', relu='adaptive') if bound_opts is None else bound_opts,
                         device, checkpoint_save_path, checkpoint_save_interval, population_bn)
        self.optimizer = GLSOptimizer(self.bounded_model, optimizer_func, lr, estimator,
                                      samples, sigma, sigma_lr, gradient_clip)
        self.lr_scheduler = lr_scheduler_func(self.optimizer, **decay)
        self.cert_train_method = 'gls'
        self.gls_options = dict(estimator=estimator, samples=samples, sigma=sigma, sigma_lr=sigma_lr, gradient_clip=gradient_clip)

    def train_model(self, train_loader, val_loader=None, start_epoch=0, end_epoch=None):
        result = gls_train_model(self.original_model, self.bounded_model, train_loader,
            self.optimizer, self.lr_scheduler, self.num_epochs, self.train_eps,
            self.n_classes, start_epoch=start_epoch, end_epoch=end_epoch,
            val_loader=val_loader, population_bn=self.population_bn,
            results_path=self.checkpoint_path, checkpoint_save_interval=self.checkpoint_save_interval,
            device=self.device, eval_eps=self.eps, **self.options)
        self.epoch = self.num_epochs if end_epoch is None else end_epoch
        return result

    def _hpo_runner(self, config, seed, epochs, train_loader, val_loader, output_dir,
                    cert_eval_samples=1000, nat_loss_weight=1, adv_loss_weight=1,
                    cert_loss_weight=1, complete_verify=False):
        seed_ctrain(seed)
        options = dict(self.options, **self.gls_options)
        for key in ('warm_up_epochs', 'ramp_up_epochs', 'shi_reg_weight', 'shi_reg_decay',
                    'l1_reg_weight', 'sigma'):
            options[key] = config[key]
        wrapper = GLSModelWrapper(copy.deepcopy(self.original_model), self.input_shape,
            self.eps, epochs, train_eps_factor=config['train_eps_factor'],
            optimizer_func=self._optimizer_from_config(config), lr=config['learning_rate'],
            lr_decay_kwargs=self._lr_decay_kwargs_from_config(config, epochs),
            bound_opts=self.bound_opts, device=self.device, population_bn=self.population_bn,
            **options)
        wrapper.train_model(train_loader, val_loader)
        config_hash = get_config_hash(config, 32)
        torch.save(wrapper.state_dict(), f'{output_dir}/nets/{config_hash}.pt')
        wrapper.eval()
        nat, cert, adv = self._evaluate_hpo_model(wrapper, val_loader, cert_eval_samples,
                                                output_dir, config_hash, complete_verify)
        return -(nat_loss_weight * nat + cert_loss_weight * cert + adv_loss_weight * adv), {
            'nat_acc': nat, 'cert_acc': cert, 'adv_acc': adv}
