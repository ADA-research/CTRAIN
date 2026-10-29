"""Public IBP-R wrapper; training mechanics live in the library trainer."""
import math
import copy

import torch
from smac.utils.configspace import get_config_hash
from CTRAIN.util import seed_ctrain

from CTRAIN.model_wrappers.certified.model_wrapper import CTRAINWrapper
from CTRAIN.train.certified.ibp_r import ibp_r_train_model
from CTRAIN.train.certified.losses.ibp_r import validate_ibp_r_options


class IBPRModelWrapper(CTRAINWrapper):
    """Wrap a classifier for adversarial training with IBP regularisation.

    IBP-R penalises the relaxation areas of unstable ReLUs to encourage tighter
    bounds. The trainer combines this penalty with adversarial cross-entropy
    and a scheduled transition from clean to adversarial training.

    See De Palma et al. (2022): https://arxiv.org/abs/2206.14772.

    ``train_eps_factor`` is the paper's alpha: it enlarges BOTH the training
    IBP and attack boxes; evaluation still uses ``eps``. ``rho`` is the paper's
    coefficient, so a published rho/2 of 1e-4 means rho=2e-4 here.
    Shi initialisation and regularisation use the common CTRAIN defaults.
    The remaining defaults use the paper's 2/255 hyperparameters. PGD step size is
    relative to the largest clipped box half-width, following CTRAIN's attack.
    With validation, select by adversarial accuracy at the original epsilon.
    """

    def __init__(self, model, input_shape, eps, num_epochs=800,
                 train_eps_factor=2.1, optimizer_func=torch.optim.SGD, lr=.01,
                 warm_up_epochs=0, ramp_up_epochs=600,
                 lr_scheduler_func=torch.optim.lr_scheduler.ExponentialLR,
                 lr_decay_kwargs=None, gradient_clip=10, l1_reg_weight=2e-5,
                 rho=2e-4, regularisation_mask=False, pgd_steps=8, pgd_alpha=.25,
                 pgd_restarts=1, checkpoint_save_path=None,
                 checkpoint_save_interval=10, bound_opts=None, device="cuda",
                 population_bn=False, relu_upper_retention=1.0, pgd_eps_factor=1.0,
                 shi_reg_weight=.5, shi_reg_decay=True, shi_init=True,
                 pgd_early_stopping=False, pgd_alpha_decay_factor=.1,
                 pgd_decay_milestones=(), lr_decay_after_ramp_up=False):
        """Configure the adversarial objective, relaxation regulariser and training schedule.

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
            l1_reg_weight (float): Weight of the L1 parameter regulariser.
            rho (float): Nonnegative weight of the unstable-ReLU relaxation-area regulariser.
            regularisation_mask (bool): Whether to apply the relaxation penalty only to correctly
                classified adversarial examples.
            pgd_steps (int): Number of projected gradient updates in the training attack.
            pgd_alpha (float): PGD step-size factor relative to the search-region radius.
            pgd_restarts (int): Number of random starts for the training attack.
            checkpoint_save_path (str or pathlib.Path, optional): Optional directory for epoch checkpoints containing the
                training trajectory.
            checkpoint_save_interval (int): Save an epoch checkpoint at this interval and at the end
                of the training call.
            bound_opts (dict, optional): Bound-propagation options passed to auto_LiRPA.
            device (str or torch.device): Device on which to run the model and tensor computations.
            population_bn (bool): Whether to recalibrate BatchNorm on training data after each epoch,
                before model selection and saving.
            relu_upper_retention (float): Fraction of unstable ReLU upper bounds retained during
                training, in (0, 1]. Values below one give an unsound training
                surrogate.
            pgd_eps_factor (float): Positive factor scaling the PGD search range, independently of the
                nominal bound region.
            shi_reg_weight (float): Weight of the interval-stability regulariser proposed by Shi et al.
                (2021).
            shi_reg_decay (bool): Whether to reduce the Shi regulariser as the perturbation radius
                increases.
            shi_init (bool): Whether to apply the initialisation proposed by Shi et al. at the start of
                training.
            pgd_early_stopping (bool): Whether to stop searching an input once a counterexample has
                been found.
            pgd_alpha_decay_factor (float): Multiplier applied to the PGD step size at each decay
                milestone.
            pgd_decay_milestones (Sequence[int]): Attack iterations at which to reduce the PGD step size.
            lr_decay_after_ramp_up (bool): Whether to delay learning-rate decay until the perturbation
                ramp has completed.
        """
        validate_ibp_r_options(rho, pgd_steps, pgd_restarts, pgd_alpha,
                              relu_upper_retention, pgd_eps_factor,
                              pgd_alpha_decay_factor, pgd_decay_milestones)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in
               (num_epochs, warm_up_epochs, ramp_up_epochs, checkpoint_save_interval)):
            raise ValueError("epoch counts and checkpoint_save_interval must be integers")
        if not math.isfinite(train_eps_factor) or train_eps_factor < 1:
            raise ValueError("IBP-R train_eps_factor must be finite and >= 1")
        if not math.isfinite(eps) or eps < 0:
            raise ValueError("eps must be nonnegative and finite")
        if num_epochs <= 0 or warm_up_epochs < 0 or ramp_up_epochs < 0 or warm_up_epochs + ramp_up_epochs > num_epochs:
            raise ValueError("warm-up and ramp-up must be nonnegative and fit num_epochs")
        if l1_reg_weight is not None and (not math.isfinite(l1_reg_weight) or l1_reg_weight < 0):
            raise ValueError("l1_reg_weight must be nonnegative and finite")
        if checkpoint_save_interval <= 0:
            raise ValueError("checkpoint_save_interval must be positive")
        if not math.isfinite(shi_reg_weight) or shi_reg_weight < 0:
            raise ValueError("shi_reg_weight must be nonnegative and finite")
        super().__init__(
            model=model, input_shape=input_shape, eps=eps,
            train_eps_factor=train_eps_factor, optimizer_func=optimizer_func, lr=lr,
            lr_scheduler_func=lr_scheduler_func,
            lr_decay_kwargs={"gamma": .95} if lr_decay_kwargs is None else lr_decay_kwargs,
            bound_opts={"conv_mode": "patches", "relu": "adaptive"} if bound_opts is None else bound_opts,
            device=device, checkpoint_save_path=checkpoint_save_path,
            checkpoint_save_interval=checkpoint_save_interval, population_bn=population_bn)
        self.cert_train_method = "ibp_r"
        self.num_epochs = num_epochs
        self.lr = lr
        self.shi_reg_weight = shi_reg_weight
        self.shi_reg_decay = shi_reg_decay
        self.shi_init = shi_init
        self.pgd_early_stopping = pgd_early_stopping
        self.pgd_alpha_decay_factor = pgd_alpha_decay_factor
        self.pgd_decay_milestones = pgd_decay_milestones
        self.lr_decay_after_ramp_up = lr_decay_after_ramp_up
        self.warm_up_epochs = warm_up_epochs
        self.ramp_up_epochs = ramp_up_epochs
        self.gradient_clip = gradient_clip
        self.l1_reg_weight = l1_reg_weight
        self.rho = rho
        self.regularisation_mask = regularisation_mask
        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha
        self.pgd_restarts = pgd_restarts
        self.pgd_eps_factor = pgd_eps_factor
        self.relu_upper_retention = relu_upper_retention

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
        trained = ibp_r_train_model(
            original_model=self.original_model, hardened_model=self.bounded_model,
            train_loader=train_loader, start_epoch=start_epoch, end_epoch=end_epoch,
            num_epochs=self.num_epochs, eps=self.train_eps,
            warm_up_epochs=self.warm_up_epochs, ramp_up_epochs=self.ramp_up_epochs,
            optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
            gradient_clip=self.gradient_clip, l1_regularisation_weight=self.l1_reg_weight,
            rho=self.rho, regularisation_mask=self.regularisation_mask,
            pgd_steps=self.pgd_steps, pgd_alpha=self.pgd_alpha, pgd_restarts=self.pgd_restarts,
            pgd_eps_factor=self.pgd_eps_factor, relu_upper_retention=self.relu_upper_retention,
            population_bn=self.population_bn, results_path=self.checkpoint_path,
            checkpoint_save_interval=self.checkpoint_save_interval, device=self.device,
            shi_regularisation_weight=self.shi_reg_weight, shi_reg_decay=self.shi_reg_decay,
            shi_init=self.shi_init, val_loader=val_loader, validation_eps=self.eps,
            n_classes=self.n_classes, lr_decay_after_ramp_up=self.lr_decay_after_ramp_up,
            pgd_early_stopping=self.pgd_early_stopping,
            pgd_decay_factor=self.pgd_alpha_decay_factor,
            pgd_decay_checkpoints=self.pgd_decay_milestones, selection_state=self.selection_state, history=self.history)
        self.epoch = self.num_epochs if end_epoch is None else end_epoch
        return trained

    def _hpo_runner(self, config, seed, epochs, train_loader, val_loader, output_dir,
                    cert_eval_samples=1000, nat_loss_weight=1, adv_loss_weight=1,
                    cert_loss_weight=1, complete_verify=False):
        """Train and evaluate one hyperparameter configuration.

        Construct a method-specific wrapper, train it and save its inference weights.
        Return the negative weighted accuracy and the natural, certified and
        adversarial validation accuracies for the optimisation procedure.
        """
        seed_ctrain(seed)
        config_hash = get_config_hash(config, 32)
        wrapper = IBPRModelWrapper(
            copy.deepcopy(self.original_model), self.input_shape, self.eps, epochs,
            train_eps_factor=config["train_eps_factor"],
            optimizer_func=self._optimizer_from_config(config), lr=config["learning_rate"],
            warm_up_epochs=config["warm_up_epochs"], ramp_up_epochs=config["ramp_up_epochs"],
            lr_scheduler_func=torch.optim.lr_scheduler.MultiStepLR,
            lr_decay_kwargs=self._lr_decay_kwargs_from_config(config, epochs),
            gradient_clip=self.gradient_clip, l1_reg_weight=config["l1_reg_weight"],
            shi_reg_weight=config["shi_reg_weight"], shi_reg_decay=config["shi_reg_decay"],
            shi_init=self.shi_init, rho=config["ibp_r:rho"],
            regularisation_mask=config["ibp_r:regularisation_mask"],
            pgd_steps=config["ibp_r:pgd_steps"], pgd_alpha=config["ibp_r:pgd_alpha"],
            pgd_restarts=config["ibp_r:pgd_restarts"],
            pgd_eps_factor=config["ibp_r:pgd_eps_factor"],
            pgd_early_stopping=self.pgd_early_stopping,
            pgd_alpha_decay_factor=self.pgd_alpha_decay_factor,
            pgd_decay_milestones=self.pgd_decay_milestones,
            lr_decay_after_ramp_up=self.lr_decay_after_ramp_up,
            population_bn=self.population_bn, relu_upper_retention=self.relu_upper_retention,
            bound_opts=self.bound_opts, device=self.device)
        wrapper.train_model(train_loader)
        torch.save(wrapper.state_dict(), f"{output_dir}/nets/{config_hash}.pt")
        wrapper.eval()
        nat, cert, adv = self._evaluate_hpo_model(
            wrapper, val_loader, cert_eval_samples, output_dir, config_hash, complete_verify)
        return -(nat_loss_weight * nat + cert_loss_weight * cert + adv_loss_weight * adv), {
            "nat_acc": nat, "cert_acc": cert, "adv_acc": adv}
