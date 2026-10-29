"""LiRPA-based set training; one update per logical loader batch."""
from CTRAIN.train.certified.util import (
    prepare_model_selection, update_model_selection, restore_model_selection,
    synchronise_bn, validate_training_run, checkpoint_due, resolve_validation_eps, EpochMetrics,
)
import math
import time
from pathlib import Path
import torch
from torch import nn
from CTRAIN.bound.set_based_zono import sequential_layers
from CTRAIN.eval import eval_certified
from CTRAIN.train.certified.losses.set_based import get_set_based_loss, validate_set_options
from CTRAIN.train.certified.progress import progress_bar, update_progress
from CTRAIN.util.util import preserve_model_state, loader_limits, loader_radius
from CTRAIN.util import save_checkpoint
from CTRAIN.train.certified.eps_scheduler import linear_schedule_fraction


def clean_bn_statistics(model, data, layers):
    """Compute differentiable clean-batch statistics for set-based propagation.

    The nominal forward pass supplies biased means and variances for each
    BatchNorm layer. Detached running-buffer updates retain 99% of previous
    statistics, while the returned batch statistics retain training gradients.
    """
    statistics = {}
    for layer in layers:
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            dims = (0, *range(2, data.ndim))
            if layer.training:
                mean, variance = data.mean(dims), data.var(dims, unbiased=False)
                with torch.no_grad():
                    layer.running_mean.mul_(.99).add_(mean, alpha=.01)
                    layer.running_var.mul_(.99).add_(variance, alpha=.01)
                    layer.num_batches_tracked.add_(1)
            else:
                mean, variance = layer.running_mean.clone(), layer.running_var.clone()
            statistics[layer] = (mean, variance)
            shape = (1, -1, *([1] * (data.ndim - 2)))
            data = (data - mean.reshape(shape)) * torch.rsqrt(variance + layer.eps).reshape(shape)
            if layer.affine:
                data = data * layer.weight.reshape(shape) + layer.bias.reshape(shape)
        else:
            data = layer(data)
    return statistics


def scheduled_epsilon(eps, epoch, batch, batches, warm_up_epochs, ramp_up_epochs):
    """Return the linearly ramped raw-input radius at the end of the requested batch."""
    return eps * linear_schedule_fraction(epoch, batch, batches, warm_up_epochs,
                                          ramp_up_epochs, at_batch_end=True)


def set_based_train_model(model, train_loader, optimizer, num_epochs, eps,
                          val_loader=None, eval_eps=None, warm_up_epochs=1,
                          ramp_up_epochs=20, lr_scheduler=None, tau=.1,
                          input_set='box', num_generators=8, error_propagation='interval',
                          propagation_batch_size=1, pgd_eps_factor=1., population_bn=False,
                          gradient_clip=10., start_epoch=0, end_epoch=None,
                          results_path=None, checkpoint_save_interval=10, device='cpu',
                          selection_state=None, *, validation_eps=None,
                          lr_decay_schedule_unit="epoch", history=None):
    """Train a classifier through centres and correlated output enclosures.

    After clean warm-up, a linear radius ramp introduces the set-based loss.
    Propagation chunks accumulate sample-weighted gradients before one optimiser
    update per logical loader batch. Clean chunk statistics are reused for
    BatchNorm propagation. Validation selects full-box forward-LiRPA accuracy
    after ramp-up; without validation, the final model is returned.

    Returns:
        (torch.nn.Module): Trained model, with selected inference weights when validation is enabled.

    Args:
        model (torch.nn.Module): PyTorch model to train or evaluate.
        train_loader (torch.utils.data.DataLoader): Training loader providing input and target batches and any required
            normalisation metadata.
        optimizer (torch.optim.Optimizer): Optimiser holding the model parameters and current update state.
        num_epochs (int): Total training budget, including warm-up and ramp-up.
        eps (float or torch.Tensor): Target perturbation radius in raw input units, before loader normalisation.
        val_loader (torch.utils.data.DataLoader, optional): Optional validation loader used for model selection. Without it,
            training returns the final weights.
        eval_eps (float or torch.Tensor, optional): Legacy name for validation_eps. Supplying conflicting values is rejected.
        warm_up_epochs (int): Initial epochs using only the natural task loss.
        ramp_up_epochs (int): Epochs over which the perturbation radius increases to its training
            target.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Learning-rate scheduler associated with the supplied optimiser.
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
        population_bn (bool): Whether to recalibrate BatchNorm on training data after each epoch,
            before model selection and saving.
        gradient_clip (float): Maximum parameter-gradient norm. None disables clipping where
            supported.
        start_epoch (int): Zero-based starting epoch in the full training schedule.
        end_epoch (int, optional): Exclusive ending epoch. None runs to the end of the configured schedule.
        results_path (str or pathlib.Path, optional): Directory for the results or checkpoints written by this function.
        checkpoint_save_interval (int): Save an epoch checkpoint at this interval and at the end
            of the training call.
        device (str or torch.device): Device on which to run the model and tensor computations.
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
    validate_set_options(tau, input_set, num_generators, error_propagation, propagation_batch_size, pgd_eps_factor)
    if not math.isfinite(eps) or eps < 0 or (validation_eps is not None and (not math.isfinite(validation_eps) or validation_eps < 0)):
        raise ValueError('Training and evaluation radii must be finite and nonnegative')
    end_epoch = validate_training_run(train_loader, num_epochs, start_epoch, end_epoch,
                                      checkpoint_save_interval, lr_decay_schedule_unit, history=history)
    for name, value in [('warm_up_epochs', warm_up_epochs), ('ramp_up_epochs', ramp_up_epochs)]:
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if gradient_clip is not None and (not math.isfinite(gradient_clip) or gradient_clip <= 0):
        raise ValueError('gradient_clip must be positive and finite or None')
    layers = sequential_layers(model)
    selection = prepare_model_selection(selection_state, val_loader, validation_eps)
    output_path = None if results_path is None else Path(results_path)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    with preserve_model_state(model):
        model.train()
        for epoch in range(start_epoch, end_epoch):
            metrics = EpochMetrics()
            epoch_start = last_update = time.monotonic()
            with progress_bar(train_loader, epoch, num_epochs, scheduled_epsilon(eps, epoch, 0, len(train_loader), warm_up_epochs, ramp_up_epochs), method="SetBased") as progress:
                for batch, (data, target) in enumerate(progress):
                    data, target = data.to(device), target.to(device)
                    current_eps = scheduled_epsilon(eps, epoch, batch, len(train_loader), warm_up_epochs, ramp_up_epochs)
                    radius = loader_radius(current_eps, train_loader, data)
                    limits = loader_limits(train_loader, data)
                    optimizer.zero_grad(set_to_none=True)
                    # Warm-up uses the logical batch; robust BN statistics are chunk-local.
                    chunk_size = len(data) if current_eps == 0 else propagation_batch_size
                    batch_loss = 0.
                    for start in range(0, len(data), chunk_size):
                        chunk = data[start:start + chunk_size]
                        statistics = clean_bn_statistics(model, chunk, layers)
                        loss = get_set_based_loss(model, chunk, target[start:start + len(chunk)], current_eps,
                                                  radius, *limits, tau=tau, input_set=input_set,
                                                  num_generators=num_generators, error_propagation=error_propagation,
                                                  pgd_eps_factor=pgd_eps_factor, bn_statistics=statistics,
                                                  layers=layers)
                        (loss * (len(chunk) / len(data))).backward()
                        batch_loss += loss.item() * len(chunk) / len(data)
                        if time.monotonic() - last_update >= 2:
                            progress.set_postfix(eps=f"{current_eps:.5g}", chunk=f"{start + len(chunk)}/{len(data)}", loss=f"{loss.item():.4f}")
                            last_update = time.monotonic()
                    if gradient_clip is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()
                    metrics.update(len(data), loss=batch_loss)
                    update_progress(progress, metrics.mean('loss'), lr=optimizer.param_groups[0]["lr"])
                    if lr_scheduler is not None and lr_decay_schedule_unit == 'batch':
                        lr_scheduler.step()
            progress.write(f"SetBased Epoch {epoch + 1}/{num_epochs}: loss={metrics.mean('loss'):.4f}, eps={current_eps:.5g}, seconds={time.monotonic() - epoch_start:.1f}")
            if population_bn:
                synchronise_bn(model, model, train_loader, device, population=True)
            if lr_scheduler is not None and lr_decay_schedule_unit == 'epoch':
                lr_scheduler.step()
            if val_loader is not None and epoch + 1 >= warm_up_epochs + ramp_up_epochs:
                progress.write("Evaluating full-box forward accuracy...")
                radius = loader_radius(eps if validation_eps is None else validation_eps, val_loader, data)
                accuracy, _ = eval_certified(model, val_loader, radius, n_classes=None,
                                             method='FORWARD', device=device,
                                             propagation_batch_size=propagation_batch_size)
                update_model_selection(selection, model, accuracy, epoch + 1)
            if history is not None:
                history.append(dict(epoch=epoch + 1, loss=metrics.mean('loss'), eps=current_eps,
                                    lr=optimizer.param_groups[0]['lr']))
            if output_path is not None and checkpoint_due(epoch + 1, end_epoch, checkpoint_save_interval):
                save_checkpoint(model, optimizer, metrics.mean('loss'), epoch + 1, output_path,
                                lr_scheduler=lr_scheduler, selection_state=selection,
                                extra_state=None if history is None else {'history': history},
                                filename=f'checkpoint_{epoch + 1}.pt')
        if val_loader is not None:
            restore_model_selection(selection, model)
    return model
