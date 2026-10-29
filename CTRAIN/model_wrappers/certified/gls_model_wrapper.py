"""Gaussian loss smoothing with existing auto_LiRPA certified losses."""
import copy
import math
import torch
from smac.utils.configspace import get_config_hash
from CTRAIN.model_wrappers.certified.model_wrapper import CTRAINWrapper
from CTRAIN.train.certified.gls import GLSOptimizer, gls_train_model
from CTRAIN.util import seed_ctrain


class GLSModelWrapper(CTRAINWrapper):
    """Wrap a classifier for certified training with Gaussian loss smoothing.

    Gaussian smoothing makes objectives based on tight relaxations suitable for
    training. RGS averages gradients at sampled parameters, while PGPE estimates
    updates for both the parameter centre and the Gaussian scale. The selected
    bound_method supplies the underlying certified classification loss.

    Following Balauca et al. (2025), PGPE experiments may start from adversarially
    pretrained weights: load them into model and set shi_init=False. Samples are
    evaluated sequentially. See https://arxiv.org/abs/2403.07095.
    """
    def __init__(self, model, input_shape, eps, num_epochs, train_eps_factor=1,
                 optimizer_func=torch.optim.Adam, lr=5e-5, warm_up_epochs=1,
                 ramp_up_epochs=80, lr_scheduler_func=torch.optim.lr_scheduler.MultiStepLR,
                 lr_decay_kwargs=None, checkpoint_save_path=None,
                 checkpoint_save_interval=10, bound_opts=None, device='cuda',
                 population_bn=False, estimator='rgs', samples=2, sigma=1e-3,
                 sigma_lr=1e-3, sigma_decay_milestones=(), sigma_decay_factor=.4,
                 bound_method='CROWN', shi_init=True, shi_reg_weight=0.,
                 shi_reg_decay=True, l1_reg_weight=0., gradient_clip=None, *, lr_decay_schedule_unit="epoch"):
        """Configure Gaussian smoothing, the certified loss and the training schedule.

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
            checkpoint_save_path (str or pathlib.Path, optional): Optional directory for epoch checkpoints containing the
                training trajectory.
            checkpoint_save_interval (int): Save an epoch checkpoint at this interval and at the end
                of the training call.
            bound_opts (dict, optional): Bound-propagation options passed to auto_LiRPA.
            device (str or torch.device): Device on which to run the model and tensor computations.
            population_bn (bool): Whether to recalibrate BatchNorm on training data after each epoch,
                before model selection and saving.
            estimator (str): Gaussian smoothing estimator: 'rgs' or 'pgpe'.
            samples (int): Number of sampled loss evaluations per update; PGPE requires an even count.
            sigma (float): Initial Gaussian perturbation scale in parameter space.
            sigma_lr (float): Learning rate for the scale optimiser used by PGPE.
            sigma_decay_milestones (Sequence[int]): Completed epochs at which RGS reduces its fixed perturbation
                scale.
            sigma_decay_factor (float): Multiplier for the RGS scale at each decay milestone.
            bound_method (str): Bound-propagation method used to construct the training loss.
            shi_init (bool): Whether to apply the initialisation proposed by Shi et al. at the start of
                training.
            shi_reg_weight (float): Weight of the interval-stability regulariser proposed by Shi et al.
                (2021).
            shi_reg_decay (bool): Whether to reduce the Shi regulariser as the perturbation radius
                increases.
            l1_reg_weight (float): Weight of the L1 parameter regulariser.
            gradient_clip (float, optional): Maximum parameter-gradient norm. None disables clipping where
                supported.
            lr_decay_schedule_unit (str): Whether to advance the learning-rate scheduler per 'epoch'
                or per 'batch'.
        """
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
        if lr_decay_schedule_unit not in ('epoch', 'batch'):
            raise ValueError("lr_decay_schedule_unit must be 'epoch' or 'batch'")
        self.num_epochs = num_epochs
        self.options = dict(lr_decay_schedule_unit=lr_decay_schedule_unit, warm_up_epochs=warm_up_epochs, ramp_up_epochs=ramp_up_epochs,
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
            (auto_LiRPA.BoundedModule): Trained model. Validation-selected inference weights are
                restored when a validation loader is supplied; otherwise final weights
                are returned. Optimiser and scheduler state remain at the final epoch.
        """
        result = gls_train_model(self.original_model, self.bounded_model, train_loader,
            self.optimizer, self.lr_scheduler, self.num_epochs, self.train_eps,
            self.n_classes, start_epoch=start_epoch, end_epoch=end_epoch,
            val_loader=val_loader, population_bn=self.population_bn,
            results_path=self.checkpoint_path, checkpoint_save_interval=self.checkpoint_save_interval,
            device=self.device, validation_eps=self.eps, selection_state=self.selection_state, history=self.history, **self.options)
        self.epoch = self.num_epochs if end_epoch is None else end_epoch
        return result

    def _hpo_runner(self, config, seed, epochs, train_loader, val_loader, output_dir,
                    cert_eval_samples=1000, nat_loss_weight=1, adv_loss_weight=1,
                    cert_loss_weight=1, complete_verify=False):
        """Train and evaluate one hyperparameter configuration.

        Construct a method-specific wrapper, train it and save its inference weights.
        Return the negative weighted accuracy and the natural, certified and
        adversarial validation accuracies for the optimisation procedure.
        """
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
