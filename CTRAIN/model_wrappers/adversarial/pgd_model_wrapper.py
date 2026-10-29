"""Adversarial teachers train as native models; LiRPA conversion is evaluation-only."""
import copy
import math

import torch
from torch import nn

from CTRAIN.model_wrappers.certified.model_wrapper import CTRAINWrapper
from CTRAIN.train.adversarial import pgd_train_model
from CTRAIN.train.adversarial.pgd import validate_pgd_training_options, make_pgd_lr_scheduler


class PGDModelWrapper(nn.Module):
    """Native PGD teacher training with the CC-Dist triangular SGD schedule.

    The optimizer is SGD(momentum=.9), peaking at ``lr`` halfway through training.
    Training accepts native PyTorch classifiers, including residual networks.
    Evaluation converts a copy using the existing CTRAIN evaluation interfaces.
    Checkpoints store native weights and batch LR state, not RNG/loader state.
    """
    def __init__(self, model, input_shape, eps, num_epochs, lr=.2,
                 pgd_steps=10, pgd_alpha=.25, pgd_restarts=1,
                 gradient_clip=10, l1_reg_weight=0., shi_init=True,
                 population_bn=False, checkpoint_save_path=None,
                 checkpoint_save_interval=10, device='cuda'):
        super().__init__()
        if type(num_epochs) is not int or num_epochs < 1:
            raise ValueError('num_epochs must be a positive integer')
        validate_pgd_training_options(eps, pgd_steps, pgd_alpha, pgd_restarts,
                                      gradient_clip, l1_reg_weight)
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError('lr must be finite and positive')
        self.original_model = model.to(device)
        self.input_shape, self.eps, self.num_epochs = input_shape, eps, num_epochs
        self.device = torch.device(device)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9)
        self.lr_scheduler = None
        self.epoch = 0
        self.options = dict(pgd_steps=pgd_steps, pgd_alpha=pgd_alpha,
            pgd_restarts=pgd_restarts, gradient_clip=gradient_clip,
            l1_reg_weight=l1_reg_weight, shi_init=shi_init, population_bn=population_bn,
            results_path=checkpoint_save_path, checkpoint_save_interval=checkpoint_save_interval)

    def forward(self, data):
        return self.original_model(data)

    def state_dict(self, *args, **kwargs):
        return self.original_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.original_model.load_state_dict(state_dict, strict=strict)

    def train_model(self, train_loader, val_loader=None, start_epoch=0, end_epoch=None):
        if val_loader is not None:
            raise ValueError('PGD teacher training returns final weights; validation selection is unsupported')
        if self.lr_scheduler is None:
            if start_epoch:
                raise ValueError('Resume from a trajectory checkpoint including its LR scheduler')
            self.lr_scheduler = make_pgd_lr_scheduler(self.optimizer, self.num_epochs, len(train_loader))
        result = pgd_train_model(self.original_model, train_loader, self.optimizer,
            self.num_epochs, self.eps, self.lr_scheduler, start_epoch, end_epoch,
            device=self.device, **self.options)
        self.epoch = self.num_epochs if end_epoch is None else end_epoch
        return result

    def resume_from_checkpoint(self, path, train_loader, end_epoch=None):
        checkpoint = torch.load(path, map_location=self.device)
        if 'lr_scheduler_state_dict' not in checkpoint:
            raise ValueError('PGD checkpoint has no LR scheduler state')
        self.original_model.load_state_dict(checkpoint['model_state_dict'])
        self.lr_scheduler = make_pgd_lr_scheduler(self.optimizer, self.num_epochs, len(train_loader))
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        return self.train_model(train_loader, start_epoch=checkpoint['epoch'], end_epoch=end_epoch)

    def _evaluation_wrapper(self):
        wrapper = CTRAINWrapper(copy.deepcopy(self.original_model), self.eps,
                                self.input_shape, device=self.device)
        wrapper.train(self.training)
        return wrapper

    def evaluate(self, *args, **kwargs):
        return self._evaluation_wrapper().evaluate(*args, **kwargs)

    def evaluate_complete(self, *args, **kwargs):
        return self._evaluation_wrapper().evaluate_complete(*args, **kwargs)
