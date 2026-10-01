"""Set-based wrapper using auto_LiRPA forward propagation operators."""
import copy
import math
from pathlib import Path
import torch
from torch import nn
from CTRAIN.model_wrappers.model_wrapper import CTRAINWrapper
from CTRAIN.bound.zonotope import sequential_layers
from CTRAIN.train.certified.set_based import set_based_train_model
from CTRAIN.train.certified.losses.set_based import validate_set_options
from CTRAIN.util.util import preserve_model_state


class SetBasedModelWrapper(CTRAINWrapper):
    def __init__(self, model, input_shape, eps, num_epochs, train_eps_factor=1.,
                 optimizer_func=torch.optim.Adam, lr=.0005, warm_up_epochs=1,
                 ramp_up_epochs=20, lr_scheduler_func=torch.optim.lr_scheduler.MultiStepLR,
                 lr_decay_kwargs=None, gradient_clip=10., checkpoint_save_path=None,
                 checkpoint_save_interval=10, bound_opts=None, device='cuda',
                 population_bn=False, tau=.1, input_set='box', num_generators=8,
                 error_propagation='interval', propagation_batch_size=1, pgd_eps_factor=1.):
        validate_set_options(tau, input_set, num_generators, error_propagation, propagation_batch_size, pgd_eps_factor)
        if not math.isfinite(eps) or eps < 0 or not math.isfinite(train_eps_factor) or train_eps_factor <= 0 or not math.isfinite(eps * train_eps_factor):
            raise ValueError('eps must be nonnegative; train_eps_factor must be positive and finite')
        for name, value, minimum in [('num_epochs', num_epochs, 1), ('warm_up_epochs', warm_up_epochs, 0),
                                      ('ramp_up_epochs', ramp_up_epochs, 0), ('checkpoint_save_interval', checkpoint_save_interval, 1)]:
            if type(value) is not int or value < minimum:
                raise ValueError(f'Invalid {name}')
        if not math.isfinite(lr) or lr <= 0 or (gradient_clip is not None and (not math.isfinite(gradient_clip) or gradient_clip <= 0)):
            raise ValueError('lr and gradient_clip must be positive and finite')
        sequential_layers(model)
        # LiRPA operators receive live parameters; this model owns all BN buffers.
        nn.Module.__init__(self)
        self.original_model = model.to(device)
        self.device, self.eps, self.train_eps = torch.device(device), eps, eps * train_eps_factor
        self.input_shape = tuple(input_shape)
        parameter = next(model.parameters())
        shape = self.input_shape if len(self.input_shape) == 4 else (1, *self.input_shape)
        with preserve_model_state(model), torch.no_grad():
            model.eval()
            self.n_classes = model(torch.zeros(shape, device=device, dtype=parameter.dtype)).shape[1]
        self.cert_train_method = 'set_based'
        self.num_epochs, self.warm_up_epochs, self.ramp_up_epochs = num_epochs, warm_up_epochs, ramp_up_epochs
        self.lr, self.gradient_clip, self.population_bn = lr, gradient_clip, population_bn
        self.optimizer_func, self.lr_scheduler_func = optimizer_func, lr_scheduler_func
        self.lr_decay_kwargs = dict(milestones=(50, 60), gamma=.1) if lr_decay_kwargs is None else dict(lr_decay_kwargs)
        self.bound_opts = {} if bound_opts is None else dict(bound_opts)
        self.optimizer = optimizer_func(model.parameters(), lr=lr)
        self.lr_scheduler = lr_scheduler_func(self.optimizer, **self.lr_decay_kwargs)
        self.checkpoint_path, self.checkpoint_save_interval = checkpoint_save_path, checkpoint_save_interval
        self.set_options = dict(tau=tau, input_set=input_set, num_generators=num_generators,
                                error_propagation=error_propagation, propagation_batch_size=propagation_batch_size,
                                pgd_eps_factor=pgd_eps_factor)
        for name, value in self.set_options.items():
            setattr(self, name, value)
        self.epoch, self.selection_state = 0, {}

    def train(self, mode=True):
        nn.Module.train(self, mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x):
        return self.original_model(x)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.original_model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.original_model.load_state_dict(state_dict, strict=strict)

    def parameters(self, recurse=True):
        return self.original_model.parameters(recurse=recurse)

    def train_model(self, train_loader, val_loader=None, start_epoch=0, end_epoch=None):
        result = set_based_train_model(self.original_model, train_loader, self.optimizer,
                                      self.num_epochs, self.train_eps, val_loader=val_loader, eval_eps=self.eps,
                                      warm_up_epochs=self.warm_up_epochs, ramp_up_epochs=self.ramp_up_epochs,
                                      lr_scheduler=self.lr_scheduler, gradient_clip=self.gradient_clip,
                                      population_bn=self.population_bn, start_epoch=start_epoch, end_epoch=end_epoch,
                                      results_path=self.checkpoint_path, checkpoint_save_interval=self.checkpoint_save_interval,
                                      device=self.device, selection_state=self.selection_state, **self.set_options)
        self.epoch = self.num_epochs if end_epoch is None else end_epoch
        return result

    def _verification_wrapper(self):
        # Fresh conversion exports current weights AND BatchNorm buffers to existing verifiers.
        return CTRAINWrapper(copy.deepcopy(self.original_model), self.eps, self.input_shape,
                             bound_opts=self.bound_opts, device=self.device)

    def evaluate(self, test_loader, test_samples=float('inf'), eval_method='ZONOTOPE'):
        if eval_method != 'ZONOTOPE':
            return self._verification_wrapper().evaluate(test_loader, test_samples, eval_method)
        from CTRAIN.eval.zonotope import evaluate_zonotope_model
        return evaluate_zonotope_model(self.original_model, test_loader, self.eps, test_samples,
                                        self.propagation_batch_size, self.error_propagation, self.device)

    def evaluate_complete(self, *args, **kwargs):
        return self._verification_wrapper().evaluate_complete(*args, **kwargs)

    def resume_from_checkpoint(self, checkpoint_path, train_loader, val_loader=None, end_epoch=None):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['lr_scheduler_state_dict'] is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.selection_state = checkpoint.get('selection_state', {})
        self.epoch = checkpoint['epoch']
        return self.train_model(train_loader, val_loader, self.epoch, end_epoch)

    def _recreate_for_hpo(self, config, epochs):
        return type(self)(copy.deepcopy(self.original_model), self.input_shape, self.eps, epochs,
                          train_eps_factor=config.get('train_eps_factor', self.train_eps / self.eps if self.eps else 1),
                          optimizer_func=self._optimizer_from_config(config), lr=config['learning_rate'],
                          warm_up_epochs=config['warm_up_epochs'], ramp_up_epochs=config['ramp_up_epochs'],
                          lr_decay_kwargs=dict(milestones=config['lr_milestones'], gamma=config['lr_decay_factor']),
                          gradient_clip=self.gradient_clip, population_bn=self.population_bn,
                          bound_opts=self.bound_opts, device=self.device,
                          **dict(self.set_options, tau=config.get('tau', self.tau)))

    def _hpo_runner(self, config, seed, epochs, train_loader, val_loader, output_dir,
                    cert_eval_samples=1000, nat_loss_weight=1., adv_loss_weight=1.,
                    cert_loss_weight=1., complete_verify=False):
        from smac.utils.configspace import get_config_hash
        from CTRAIN.util import seed_ctrain
        seed_ctrain(seed)
        wrapper = self._recreate_for_hpo(config, epochs)
        wrapper.train_model(train_loader, val_loader)
        config_hash = get_config_hash(config, 32)
        directory = Path(output_dir) / 'nets'
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(wrapper.state_dict(), directory / f'{config_hash}.pt')
        clean, certified, adversarial = self._evaluate_hpo_model(
            wrapper, val_loader, cert_eval_samples, output_dir, config_hash, complete_verify)
        return -(nat_loss_weight * clean + adv_loss_weight * adversarial + cert_loss_weight * certified), dict(
            nat_acc=clean, adv_acc=adversarial, cert_acc=certified)
