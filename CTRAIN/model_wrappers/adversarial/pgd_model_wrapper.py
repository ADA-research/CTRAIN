"""Adversarial teachers train as native models; LiRPA conversion is evaluation-only."""
import copy
import math

import torch
from torch import nn

from CTRAIN.model_wrappers.certified.model_wrapper import CTRAINWrapper
from CTRAIN.train.adversarial import pgd_train_model
from CTRAIN.train.adversarial.pgd import validate_pgd_training_options, make_pgd_lr_scheduler


class PGDModelWrapper(nn.Module):
    """Wrap a native PyTorch classifier for PGD adversarial training.

    This wrapper can train adversarial teachers for CC-Dist without constructing
    a bound graph. SGD with momentum 0.9 follows a triangular learning-rate
    schedule, peaking at lr halfway through training. Evaluation converts a
    copy of the trained network through the common CTRAIN interfaces.

    The optimiser is SGD(momentum=.9), peaking at ``lr`` halfway through training.
    Training accepts native PyTorch classifiers, including residual networks.
    Evaluation converts a copy using the existing CTRAIN evaluation interfaces.
    Checkpoints store native weights and batch LR state, not RNG/loader state.
    """
    def __init__(self, model, input_shape, eps, num_epochs, lr=.2,
                 pgd_steps=10, pgd_alpha=.25, pgd_restarts=1,
                 gradient_clip=10, l1_reg_weight=0., shi_init=True,
                 population_bn=False, checkpoint_save_path=None,
                 checkpoint_save_interval=10, device='cuda'):
        """Configure native PGD training with the triangular SGD schedule.

        Args:
            model (torch.nn.Module): PyTorch model to train or evaluate.
            input_shape (tuple[int, ...]): Shape of a model input, excluding the batch dimension unless supplied
                explicitly.
            eps (float or torch.Tensor): Target perturbation radius in raw input units, before loader normalisation.
            num_epochs (int): Total training budget, including warm-up and ramp-up.
            lr (float): Learning rate passed to the optimiser.
            pgd_steps (int): Number of projected gradient updates in the training attack.
            pgd_alpha (float): PGD step-size factor relative to the search-region radius.
            pgd_restarts (int): Number of random starts for the training attack.
            gradient_clip (float): Maximum parameter-gradient norm. None disables clipping where
                supported.
            l1_reg_weight (float): Weight of the L1 parameter regulariser.
            shi_init (bool): Whether to apply the initialisation proposed by Shi et al. at the start of
                training.
            population_bn (bool): Whether to recalibrate BatchNorm on training data after each epoch,
                before model selection and saving.
            checkpoint_save_path (str or pathlib.Path, optional): Optional directory for epoch checkpoints containing the
                training trajectory.
            checkpoint_save_interval (int): Save an epoch checkpoint at this interval and at the end
                of the training call.
            device (str or torch.device): Device on which to run the model and tensor computations.
        """
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
        self.selection_state = {}
        self.options = dict(pgd_steps=pgd_steps, pgd_alpha=pgd_alpha,
            pgd_restarts=pgd_restarts, gradient_clip=gradient_clip,
            l1_reg_weight=l1_reg_weight, shi_init=shi_init, population_bn=population_bn,
            results_path=checkpoint_save_path, checkpoint_save_interval=checkpoint_save_interval)

    def forward(self, data):
        """Evaluate the native network on the supplied input batch."""
        return self.original_model(data)

    def state_dict(self, *args, **kwargs):
        """Return the native network parameters and persistent buffers for inference."""
        return self.original_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Load native network parameters and buffers, checking keys when strict=True."""
        return self.original_model.load_state_dict(state_dict, strict=strict)

    def train_model(self, train_loader, val_loader=None, start_epoch=0, end_epoch=None):
        """Train the wrapped model using the configured method and schedule.

        The wrapper forwards its configuration to the corresponding library trainer.
        start_epoch and end_epoch select a segment of the full schedule, retaining
        its original warm-up, ramp-up and learning-rate milestones.

        Args:
            train_loader (torch.utils.data.DataLoader): Training loader providing input and target batches and any required
                normalisation metadata.
            val_loader (torch.utils.data.DataLoader, optional): Optional validation loader used for model selection. Without it,
                training returns the final weights.
            start_epoch (int): Zero-based starting epoch in the full training schedule.
            end_epoch (int, optional): Exclusive ending epoch. None runs to the end of the configured schedule.

        Returns:
            (torch.nn.Module): Trained model. Validation-selected inference weights are
                restored when a validation loader is supplied; otherwise final weights
                are returned. Optimiser and scheduler state remain at the final epoch.
        """
        if self.lr_scheduler is None:
            if start_epoch:
                raise ValueError('Resume from a trajectory checkpoint including its LR scheduler')
            self.lr_scheduler = make_pgd_lr_scheduler(self.optimizer, self.num_epochs, len(train_loader))
        result = pgd_train_model(self.original_model, train_loader, self.optimizer,
            self.num_epochs, self.eps, self.lr_scheduler, start_epoch, end_epoch,
            device=self.device, val_loader=val_loader, eval_eps=self.eps,
            selection_state=self.selection_state, **self.options)
        self.epoch = self.num_epochs if end_epoch is None else end_epoch
        return result

    def resume_from_checkpoint(self, path, train_loader, end_epoch=None, val_loader=None):
        """Restore native model, SGD and scheduler state, then continue PGD training.

        Args:
            path (str): Checkpoint containing the completed epoch and triangular scheduler
                state.
            train_loader (DataLoader): Training loader using the original batch protocol.
            end_epoch (int, optional): Exclusive ending epoch; None completes the schedule.
            val_loader (DataLoader, optional): Validation loader used for model selection.

        Returns:
            (torch.nn.Module): Trained native classifier.

        RNG and data-loader state are not restored by these checkpoints.
        """
        checkpoint = torch.load(path, map_location=self.device)
        if 'lr_scheduler_state_dict' not in checkpoint:
            raise ValueError('PGD checkpoint has no LR scheduler state')
        self.selection_state = checkpoint.get('selection_state', {})
        self.original_model.load_state_dict(checkpoint['model_state_dict'])
        self.lr_scheduler = make_pgd_lr_scheduler(self.optimizer, self.num_epochs, len(train_loader))
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        return self.train_model(train_loader, val_loader=val_loader, start_epoch=checkpoint['epoch'], end_epoch=end_epoch)

    def _evaluation_wrapper(self):
        """Construct a bounded copy of the native classifier in the caller's current mode."""
        wrapper = CTRAINWrapper(copy.deepcopy(self.original_model), self.eps,
                                self.input_shape, device=self.device)
        wrapper.train(self.training)
        return wrapper

    def evaluate(self, *args, **kwargs):
        """Evaluate current native weights through a fresh bounded network copy.

        Arguments and returned natural, certified and adversarial accuracies follow
        CTRAINWrapper.evaluate.
        """
        return self._evaluation_wrapper().evaluate(*args, **kwargs)

    def evaluate_complete(self, *args, **kwargs):
        """Evaluate current native weights through the complete-verification interface.

        Arguments and returned accuracy tuple follow CTRAINWrapper.evaluate_complete.
        """
        return self._evaluation_wrapper().evaluate_complete(*args, **kwargs)
