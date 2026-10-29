"""Gaussian loss smoothing (Balauca et al., arXiv:2403.07095).

The released PGPE implementation uses preconditioned score gradients: d*(L+-L-)
for the mean and (d²-s²)/s*(mean(L+,L-)-baseline) for the deviation.
RGS averages ordinary gradients at independent Gaussian parameter samples.
"""
from CTRAIN.train.certified.util import (
    prepare_model_selection, update_model_selection, restore_model_selection,
    synchronise_bn, validate_training_run, checkpoint_due, resolve_validation_eps, EpochMetrics,
)
import math
import torch
from auto_LiRPA import PerturbationLpNorm
from CTRAIN.bound import bound_ibp, bound_crown, bound_crown_ibp
from CTRAIN.train.certified.losses.util import get_loss_from_bounds
from CTRAIN.train.certified.initialisation import ibp_init_shi
from CTRAIN.train.certified.regularisers import get_shi_regulariser, get_l1_reg
from CTRAIN.train.certified.eps_scheduler import BaseScheduler, linear_schedule_fraction
from CTRAIN.util import save_checkpoint
from CTRAIN.util.util import preserve_model_state
from CTRAIN.train.certified.progress import progress_bar, update_progress, log_epoch_summary


class GLSOptimizer(torch.optim.Optimizer):
    """Optimise a Gaussian-smoothed loss using RGS or PGPE.

    RGS averages ordinary gradients at independent Gaussian parameter samples.
    PGPE uses antithetic samples and the unperturbed loss as a baseline to
    estimate centre and scale updates, following the released implementation.
    The sample count includes both sides of each PGPE pair and must be even.

    Buffers and centre weights are restored between samples. RGS keeps a fixed
    scale, optionally decayed by the trainer; PGPE learns its scale with a maximum
    2% change per update. Its optimiser states and scales can be checkpointed.
    """
    def __init__(self, model, optimizer_func=torch.optim.Adam, lr=5e-5,
                 estimator='rgs', samples=2, sigma=1e-3, sigma_lr=1e-3, gradient_clip=None):
        """Configure the smoothing estimator, parameter optimiser and Gaussian scales.

        Args:
            model (torch.nn.Module): PyTorch model to train or evaluate.
            optimizer_func (callable): Optimiser constructor accepting the model parameters and a learning
                rate.
            lr (float): Learning rate passed to the optimiser.
            estimator (str): Gaussian smoothing estimator: 'rgs' or 'pgpe'.
            samples (int): Number of sampled loss evaluations per update; PGPE requires an even count.
            sigma (float): Initial Gaussian perturbation scale in parameter space.
            sigma_lr (float): Learning rate for the scale optimiser used by PGPE.
            gradient_clip (float, optional): Maximum parameter-gradient norm. None disables clipping where
                supported.
        """
        if estimator not in ('rgs', 'pgpe'):
            raise ValueError("estimator must be 'rgs' or 'pgpe'")
        if type(samples) is not int or samples < 1 or (estimator == 'pgpe' and samples % 2):
            raise ValueError('samples must be positive and even for PGPE')
        if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(sigma_lr) or sigma_lr < 0:
            raise ValueError('sigma must be positive finite and sigma_lr nonnegative finite')
        self.gradient_clip = gradient_clip
        self.model = model
        self.estimator, self.samples = estimator, samples
        params = [p for p in model.parameters() if p.requires_grad]
        super().__init__(params, dict(lr=lr))
        self.center_optimizer = optimizer_func(params, lr=lr)
        self.param_groups = self.center_optimizer.param_groups
        self.scales = [torch.nn.Parameter(torch.full_like(p, sigma)) for p in params]
        self.scale_optimizer = optimizer_func(self.scales, lr=sigma_lr)

    def state_dict(self):
        """Return centre-optimiser, scale-optimiser and Gaussian-scale state for checkpointing."""
        return dict(center=self.center_optimizer.state_dict(),
                    scale=self.scale_optimizer.state_dict(),
                    scales=[s.detach().clone() for s in self.scales],
                    estimator=self.estimator, samples=self.samples)

    def load_state_dict(self, state):
        """Restore smoothing and optimiser state, requiring a matching estimator and sample count."""
        if state['estimator'] != self.estimator or state['samples'] != self.samples:
            raise ValueError('GLS checkpoint estimator/sample configuration differs')
        self.center_optimizer.load_state_dict(state['center'])
        self.param_groups = self.center_optimizer.param_groups
        self.scale_optimizer.load_state_dict(state['scale'])
        with torch.no_grad():
            for scale, value in zip(self.scales, state['scales']):
                scale.copy_(value)

    def step(self, closure):
        """Evaluate the closure at sampled parameters and update the Gaussian centre.

        The closure returns a scalar loss without applying an optimiser step. RGS
        computes and averages sample gradients; PGPE evaluates antithetic losses and
        also updates scale parameters. Sampling restores model parameters, buffers
        and pre-existing gradients if loss or gradient computation fails.

        Args:
            closure (callable): Zero-argument scalar objective on the current sampled model.

        Returns:
            (torch.Tensor): Mean sampled loss detached from the autograd graph.
        """
        params = [p for g in self.param_groups for p in g['params']]
        centers = [p.detach().clone() for p in params]
        buffers = [(b, b.clone()) for b in self.model.buffers()]
        old_grads = [None if p.grad is None else p.grad.clone() for p in params]
        grads = [torch.zeros_like(p) for p in params]
        sigma_grads = [torch.zeros_like(s) for s in self.scales]
        total = 0.

        def restore():
            with torch.no_grad():
                for p, center in zip(params, centers):
                    p.copy_(center)
                for buf, saved in buffers:
                    buf.copy_(saved)

        def evaluate(noise, sign, backward):
            restore()
            with torch.no_grad():
                for p, center, d in zip(params, centers, noise):
                    p.copy_(center + sign * d)
            self.zero_grad(set_to_none=True)
            with torch.enable_grad() if backward else torch.no_grad():
                loss = closure()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite GLS loss')
                if backward:
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                        raise FloatingPointError('Non-finite GLS gradients')
                    # The released RGS implementation clips each sample's
                    # gradient before averaging, not the averaged gradient.
                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(params, self.gradient_clip)
            return loss.detach()

        try:
            if self.estimator == 'pgpe':
                restore()
                with torch.no_grad():
                    baseline = closure().detach()
                    if not torch.isfinite(baseline):
                        raise FloatingPointError('Non-finite GLS baseline loss')
            count = self.samples if self.estimator == 'rgs' else self.samples // 2
            for _ in range(count):
                noise = [torch.randn_like(s) * s.detach() for s in self.scales]
                plus = evaluate(noise, 1, self.estimator == 'rgs')
                if self.estimator == 'rgs':
                    for grad, p in zip(grads, params):
                        if p.grad is not None:
                            grad.add_(p.grad / count)
                    total += plus / count
                else:
                    minus = evaluate(noise, -1, False)
                    for grad, sg, d, s in zip(grads, sigma_grads, noise, self.scales):
                        grad.add_(d * (plus - minus) / count)
                        sg.add_((d.square() - s.detach().square()) / s.detach()
                                * ((plus + minus) / 2 - baseline) / count)
                    total += (plus + minus) / (2 * count)
        except BaseException:
            for p, grad in zip(params, old_grads):
                p.grad = grad
            raise
        finally:
            restore()
        for p, grad in zip(params, grads):
            p.grad = grad
        if self.gradient_clip is not None and self.estimator == 'pgpe':
            torch.nn.utils.clip_grad_norm_(params, self.gradient_clip)
        self.center_optimizer.step()
        if self.estimator == 'pgpe':
            old_scales = [s.detach().clone() for s in self.scales]
            for s, grad in zip(self.scales, sigma_grads):
                s.grad = grad
            self.scale_optimizer.step()
            with torch.no_grad():
                for s, old in zip(self.scales, old_scales):
                    s.copy_(torch.maximum(old * .98, torch.minimum(old * 1.02, s)))
        return total


def gls_train_model(original_model, hardened_model, train_loader, optimizer,
                    lr_scheduler, num_epochs, eps, n_classes, start_epoch=0,
                    end_epoch=None, warm_up_epochs=1, ramp_up_epochs=80,
                    bound_method='CROWN', shi_init=True, shi_reg_weight=0.,
                    shi_reg_decay=True, l1_reg_weight=0., population_bn=False,
                    sigma_decay_milestones=(), sigma_decay_factor=.4,
                    results_path=None, checkpoint_save_interval=10, device='cuda',
                    val_loader=None, eval_eps=None, selection_state=None, *,
                    validation_eps=None, lr_decay_schedule_unit="epoch", history=None):
    """Train a classifier using a Gaussian-smoothed certified objective.

    After natural-loss warm-up, a linear perturbation ramp introduces the
    selected bound-based loss. GLSOptimizer evaluates this objective at sampled
    parameter values. Validation selects the centre model using sound bounds
    after ramp-up, while checkpoints retain the optimiser trajectory.

    No clean/robust loss mixing is used after warm-up. Without validation, the
    final model is returned. Validation selects by sound centre-model bounds
    after ramp-up; optimiser checkpoints retain the training trajectory.

    Args:
        original_model (torch.nn.Module): Original PyTorch model sharing the trainable network parameters.
        hardened_model (auto_LiRPA.BoundedModule): Bounded auto_LiRPA graph used for training and bound propagation.
        train_loader (torch.utils.data.DataLoader): Training loader providing input and target batches and any required
            normalisation metadata.
        optimizer (torch.optim.Optimizer): Optimiser holding the model parameters and current update state.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler): Learning-rate scheduler associated with the supplied optimiser.
        num_epochs (int): Total training budget, including warm-up and ramp-up.
        eps (float or torch.Tensor): Target perturbation radius in raw input units, before loader normalisation.
        n_classes (int): Number of output classes.
        start_epoch (int): Zero-based starting epoch in the full training schedule.
        end_epoch (int, optional): Exclusive ending epoch. None runs to the end of the configured schedule.
        warm_up_epochs (int): Initial epochs using only the natural task loss.
        ramp_up_epochs (int): Epochs over which the perturbation radius increases to its training
            target.
        bound_method (str): Bound-propagation method used to construct the training loss.
        shi_init (bool): Whether to apply the initialisation proposed by Shi et al. at the start of
            training.
        shi_reg_weight (float): Weight of the interval-stability regulariser proposed by Shi et al.
            (2021).
        shi_reg_decay (bool): Whether to reduce the Shi regulariser as the perturbation radius
            increases.
        l1_reg_weight (float): Weight of the L1 parameter regulariser.
        population_bn (bool): Whether to recalibrate BatchNorm on training data after each epoch,
            before model selection and saving.
        sigma_decay_milestones (Sequence[int]): Completed epochs at which RGS reduces its fixed perturbation
            scale.
        sigma_decay_factor (float): Multiplier for the RGS scale at each decay milestone.
        results_path (str or pathlib.Path, optional): Directory for the results or checkpoints written by this function.
        checkpoint_save_interval (int): Save an epoch checkpoint at this interval and at the end
            of the training call.
        device (str or torch.device): Device on which to run the model and tensor computations.
        val_loader (torch.utils.data.DataLoader, optional): Optional validation loader used for model selection. Without it,
            training returns the final weights.
        eval_eps (float or torch.Tensor, optional): Legacy name for validation_eps. Supplying conflicting values is rejected.
        selection_state (dict, optional): Mutable dictionary retaining the best validation score, inference
            weights and epoch across training calls.
        validation_eps (float or torch.Tensor, optional): Fixed raw-input radius used for validation; None uses the training
            target.
        lr_decay_schedule_unit (str): Whether to advance the learning-rate scheduler per 'epoch'
            or per 'batch'.
        history (list, optional): Optional list to which the trainer appends per-epoch losses, radii and
            learning rates.
    """
    validation_eps = resolve_validation_eps(validation_eps, eval_eps)
    selection = prepare_model_selection(selection_state, val_loader, validation_eps)
    bounds = {'IBP': bound_ibp, 'CROWN': bound_crown, 'CROWN-IBP': bound_crown_ibp}
    if bound_method not in bounds:
        raise ValueError('bound_method must be IBP, CROWN, or CROWN-IBP')
    end_epoch = validate_training_run(train_loader, num_epochs, start_epoch, end_epoch,
                                      checkpoint_save_interval, lr_decay_schedule_unit, history=history)
    if warm_up_epochs < 0 or ramp_up_epochs < 0 or warm_up_epochs + ramp_up_epochs > num_epochs:
        raise ValueError('invalid warm-up/ramp-up schedule')
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = BaseScheduler(num_epochs, eps, train_loader.mean,
                              train_loader.std if train_loader.normalised else torch.ones_like(train_loader.std),
                              eps_schedule=(warm_up_epochs, ramp_up_epochs),
                              eps_schedule_unit='epoch', batches_per_epoch=len(train_loader))
    with preserve_model_state(original_model, hardened_model):
        if start_epoch == 0 and end_epoch > start_epoch and shi_init:
            ibp_init_shi(original_model, hardened_model)
        for epoch in range(start_epoch, end_epoch):
            hardened_model.train()
            original_model.train()
            metrics = EpochMetrics()
            # tqdm.auto otherwise creates an extra loader iterator and consumes RNG.
            with progress_bar(iter(train_loader), epoch, num_epochs, None,
                              method=f'GLS-{optimizer.estimator.upper()}',
                              total=len(train_loader)) as progress_bar_epoch:
                for index, (data, target) in enumerate(progress_bar_epoch):
                    data, target = data.to(device), target.to(device)
                    fraction = linear_schedule_fraction(epoch, index, len(train_loader),
                                                        warm_up_epochs, ramp_up_epochs)
                    scheduler.cur_eps = eps * fraction
                    radius = scheduler.get_cur_eps().to(data).reshape(-1, 1, 1)
                    ptb = PerturbationLpNorm(norm=float('inf'), eps=radius,
                            x_L=torch.maximum(data - radius, train_loader.min.to(data)),
                            x_U=torch.minimum(data + radius, train_loader.max.to(data)))

                    def closure():
                        output = hardened_model(data)
                        if fraction:
                            lb, _ = bounds[bound_method](hardened_model, ptb, data, target, n_classes)
                            loss = get_loss_from_bounds(lb, criterion)
                        else:
                            loss = criterion(output, target)
                        if shi_reg_weight and fraction < 1:
                            bound_ibp(hardened_model, ptb, data, target, n_classes)
                            loss = loss + shi_reg_weight * get_shi_regulariser(
                                hardened_model, ptb, data, target, scheduler,
                                n_classes=n_classes, device=device, regularisation_decay=shi_reg_decay)
                        if l1_reg_weight:
                            loss = loss + l1_reg_weight * get_l1_reg(original_model, device=device)
                        return loss

                    metrics.update(len(data), loss=optimizer.step(closure))
                    # Update BN once at the centre, rather than once per noisy sample.
                    with torch.no_grad():
                        hardened_model(data)
                    update_progress(progress_bar_epoch, metrics.mean('loss'), lr=optimizer.param_groups[0]['lr'])
                    if lr_scheduler is not None and lr_decay_schedule_unit == 'batch':
                        lr_scheduler.step()
            synchronise_bn(original_model, hardened_model, train_loader, device, population_bn)
            if lr_scheduler is not None and lr_decay_schedule_unit == 'epoch':
                lr_scheduler.step()
            if optimizer.estimator == 'rgs' and epoch + 1 in sigma_decay_milestones:
                with torch.no_grad():
                    for scale in optimizer.scales:
                        scale.mul_(sigma_decay_factor)
            if val_loader is not None and fraction == 1.:
                from CTRAIN.eval import eval_certified
                eval_radius = torch.as_tensor(eps if validation_eps is None else validation_eps, device=device)
                if val_loader.normalised:
                    eval_radius = eval_radius / val_loader.std.to(device)
                score, _ = eval_certified(hardened_model, val_loader, eval_radius.reshape(-1, 1, 1),
                                          n_classes=n_classes, method=bound_method, device=device)
                update_model_selection(selection, hardened_model, score, epoch + 1)
            log_epoch_summary(epoch, num_epochs, metrics.mean('loss'), nat_acc=None)
            if history is not None:
                history.append(dict(epoch=epoch + 1, loss=metrics.mean('loss'),
                                    eps=float(scheduler.cur_eps), lr=optimizer.param_groups[0]['lr']))
            if results_path is not None and checkpoint_due(epoch + 1, end_epoch, checkpoint_save_interval):
                save_checkpoint(hardened_model, optimizer, metrics.mean('loss'), epoch + 1, results_path,
                                lr_scheduler=lr_scheduler, selection_state=selection,
                                extra_state=None if history is None else {'history': history})
        if val_loader is not None and restore_model_selection(selection, hardened_model):
            synchronise_bn(original_model, hardened_model, train_loader, device, False)
    return hardened_model
