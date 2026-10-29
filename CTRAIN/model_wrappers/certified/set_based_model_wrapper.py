"""Set-based wrapper using auto_LiRPA forward propagation operators."""
import copy
import math
from pathlib import Path
import torch
from torch import nn
from CTRAIN.model_wrappers.certified.model_wrapper import CTRAINWrapper
from CTRAIN.bound.set_based_zono import sequential_layers
from CTRAIN.train.certified.set_based import set_based_train_model
from CTRAIN.train.certified.losses.set_based import validate_set_options
from CTRAIN.util.util import preserve_model_state


class SetBasedModelWrapper(CTRAINWrapper):
    """Wrap a sequential classifier for set-based certified training.

    The objective combines classification at the output centre with a penalty
    on the propagated output enclosure. Training retains correlated generators
    through auto_LiRPA forward operators and uses clean-batch BatchNorm
    statistics. Supported layers are checked against the model's forward graph.
    Evaluation converts current native weights and running buffers to a fresh
    bounded graph; full-box certification is independent of attack-generated
    training sets.

    See Koller et al. (2025), Set-Based Training for Neural Network Verification.
    """
    def __init__(self, model, input_shape, eps, num_epochs, train_eps_factor=1.,
                 optimizer_func=torch.optim.Adam, lr=.0005, warm_up_epochs=1,
                 ramp_up_epochs=20, lr_scheduler_func=torch.optim.lr_scheduler.MultiStepLR,
                 lr_decay_kwargs=None, gradient_clip=10., checkpoint_save_path=None,
                 checkpoint_save_interval=10, bound_opts=None, device='cuda',
                 population_bn=False, tau=.1, input_set='box', num_generators=8,
                 error_propagation='interval', propagation_batch_size=1, pgd_eps_factor=1., *,
                 lr_decay_schedule_unit="epoch"):
        """Configure the input enclosure, output penalty and set-based training schedule.

        Args:
            model (torch.nn.Module): PyTorch model to train or evaluate.
            input_shape (tuple[int, ...]): Shape of a model input, excluding the batch dimension unless supplied
                explicitly.
            eps (float or torch.Tensor): Target perturbation radius in raw input units, before loader normalisation.
            num_epochs (int): Total training budget, including warm-up and ramp-up.
            train_eps_factor (float): Factor applied to the training radius; the nominal evaluation
                radius remains eps.
            optimizer_func (callable): Optimiser constructor accepting the model parameters and a learning
                rate.
            lr (float): Learning rate passed to the optimiser.
            warm_up_epochs (int): Initial epochs using only the natural task loss.
            ramp_up_epochs (int): Epochs over which the perturbation radius increases to its training
                target.
            lr_scheduler_func (callable): Constructor for the learning-rate scheduler.
            lr_decay_kwargs (dict, optional): Scheduler arguments, such as milestones and gamma for MultiStepLR.
            gradient_clip (float): Maximum parameter-gradient norm. None disables clipping where
                supported.
            checkpoint_save_path (str or pathlib.Path, optional): Optional directory for epoch checkpoints containing the
                training trajectory.
            checkpoint_save_interval (int): Save an epoch checkpoint at this interval and at the end
                of the training call.
            bound_opts (dict, optional): Bound-propagation options passed to auto_LiRPA.
            device (str or torch.device): Device on which to run the model and tensor computations.
            population_bn (bool): Whether to recalibrate BatchNorm on training data after each epoch,
                before model selection and saving.
            tau (float): Weight of the output-enclosure penalty, with 1 - tau weighting centre
                cross-entropy.
            input_set (str): Training input representation: 'box' or an attack-generated 'zonotope'.
            num_generators (int): Number of attack directions used to construct an input zonotope.
            error_propagation (str): Whether nonlinear interval errors remain an 'interval' or become
                explicit 'generators'.
            propagation_batch_size (int): Maximum number of inputs propagated together to limit
                bound-computation memory.
            pgd_eps_factor (float): Positive factor scaling the PGD search range, independently of the
                nominal bound region.
            lr_decay_schedule_unit (str): Whether to advance the learning-rate scheduler per 'epoch'
                or per 'batch'.
        """
        validate_set_options(tau, input_set, num_generators, error_propagation, propagation_batch_size, pgd_eps_factor)
        if not math.isfinite(eps) or eps < 0 or not math.isfinite(train_eps_factor) or train_eps_factor <= 0 or not math.isfinite(eps * train_eps_factor):
            raise ValueError('eps must be nonnegative; train_eps_factor must be positive and finite')
        for name, value, minimum in [('num_epochs', num_epochs, 1), ('warm_up_epochs', warm_up_epochs, 0),
                                      ('ramp_up_epochs', ramp_up_epochs, 0), ('checkpoint_save_interval', checkpoint_save_interval, 1)]:
            if type(value) is not int or value < minimum:
                raise ValueError(f'Invalid {name}')
        if not math.isfinite(lr) or lr <= 0 or (gradient_clip is not None and (not math.isfinite(gradient_clip) or gradient_clip <= 0)):
            raise ValueError('lr and gradient_clip must be positive and finite')
        if lr_decay_schedule_unit not in ('epoch', 'batch'):
            raise ValueError("lr_decay_schedule_unit must be 'epoch' or 'batch'")
        self.lr_decay_schedule_unit = lr_decay_schedule_unit
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
        self.epoch, self.selection_state, self.history = 0, {}, []

    def train(self, mode=True):
        """Set the wrapper and native network to the requested training mode."""
        nn.Module.train(self, mode)
        return self

    def eval(self):
        """Set the wrapper and native network to evaluation mode."""
        return self.train(False)

    def forward(self, x):
        """Evaluate the native network on the supplied input batch."""
        return self.original_model(x)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Return the native network parameters and persistent buffers for inference."""
        return self.original_model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        """Load native network parameters and buffers, checking keys when strict=True."""
        return self.original_model.load_state_dict(state_dict, strict=strict)

    def parameters(self, recurse=True):
        """Iterate over the native network parameters exposed to the optimiser."""
        return self.original_model.parameters(recurse=recurse)

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
        result = set_based_train_model(self.original_model, train_loader, self.optimizer,
                                      self.num_epochs, self.train_eps, val_loader=val_loader, validation_eps=self.eps,
                                      warm_up_epochs=self.warm_up_epochs, ramp_up_epochs=self.ramp_up_epochs,
                                      lr_scheduler=self.lr_scheduler, lr_decay_schedule_unit=self.lr_decay_schedule_unit, gradient_clip=self.gradient_clip,
                                      population_bn=self.population_bn, start_epoch=start_epoch, end_epoch=end_epoch,
                                      results_path=self.checkpoint_path, checkpoint_save_interval=self.checkpoint_save_interval,
                                      device=self.device, selection_state=self.selection_state, history=self.history, **self.set_options)
        self.epoch = self.num_epochs if end_epoch is None else end_epoch
        return result

    def _verification_wrapper(self):
        # Fresh conversion exports current weights AND BatchNorm buffers to existing verifiers.
        """Convert a private copy of current native weights and buffers for ordinary robustness
        evaluation.
        """
        return CTRAINWrapper(copy.deepcopy(self.original_model), self.eps, self.input_shape,
                             bound_opts=self.bound_opts, device=self.device)

    def evaluate(self, test_loader, test_samples=float('inf'), eval_method='FORWARD',
                 adversarial_method='PGD', alpha_iterations=20, autoattack_kwargs=None,
                 propagation_batch_size=None):
        """Evaluate current native weights through a fresh bounded network copy.

        Arguments and returned natural, certified and adversarial accuracies follow
        CTRAINWrapper.evaluate.
        """
        return self._verification_wrapper().evaluate(
            test_loader, test_samples, eval_method, adversarial_method,
            alpha_iterations, autoattack_kwargs,
            self.propagation_batch_size if propagation_batch_size is None else propagation_batch_size)

    def evaluate_complete(self, *args, **kwargs):
        """Evaluate current native weights through the complete-verification interface.

        Arguments and returned accuracy tuple follow CTRAINWrapper.evaluate_complete.
        """
        return self._verification_wrapper().evaluate_complete(*args, **kwargs)

    def resume_from_checkpoint(self, checkpoint_path, train_loader, val_loader=None, end_epoch=None):
        """Restore the training trajectory and selection state, without RNG replay.

        Loads model, optimiser and LR scheduler state; caller-managed RNG and
        data-loader state are not restored, as in the other training wrappers.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['lr_scheduler_state_dict'] is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.selection_state = checkpoint.get('selection_state', {})
        self.history = checkpoint.get('history', [])
        self.epoch = checkpoint['epoch']
        return self.train_model(train_loader, val_loader, self.epoch, end_epoch)

    def _recreate_for_hpo(self, config, epochs):
        """Recreate the set-based wrapper with sampled hyperparameters and the existing propagation
        options.
        """
        return type(self)(copy.deepcopy(self.original_model), self.input_shape, self.eps, epochs,
                          train_eps_factor=config.get('train_eps_factor', self.train_eps / self.eps if self.eps else 1),
                          optimizer_func=self._optimizer_from_config(config), lr=config['learning_rate'],
                          warm_up_epochs=config['warm_up_epochs'], ramp_up_epochs=config['ramp_up_epochs'],
                          lr_decay_kwargs=dict(milestones=config['lr_milestones'], gamma=config['lr_decay_factor']),
                          gradient_clip=self.gradient_clip, population_bn=self.population_bn,
                          lr_decay_schedule_unit=self.lr_decay_schedule_unit, bound_opts=self.bound_opts, device=self.device,
                          **dict(self.set_options, tau=config.get('tau', self.tau)))

    def _hpo_runner(self, config, seed, epochs, train_loader, val_loader, output_dir,
                    cert_eval_samples=1000, nat_loss_weight=1., adv_loss_weight=1.,
                    cert_loss_weight=1., complete_verify=False):
        """Train and evaluate one hyperparameter configuration.

        Construct a method-specific wrapper, train it and save its inference weights.
        Return the negative weighted accuracy and the natural, certified and
        adversarial validation accuracies for the optimisation procedure.
        """
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
