"""LiRPA-based set training; one update per logical loader batch."""
import copy
import math
import time
from pathlib import Path
import torch
from torch import nn
from CTRAIN.bound.zonotope import Zonotope, sequential_layers
from CTRAIN.eval.zonotope import eval_zonotope, loader_limits, loader_radius
from CTRAIN.train.certified.losses.set_based import get_set_based_loss, attack_zonotope, validate_set_options
from CTRAIN.train.certified.progress import progress_bar, update_progress
from CTRAIN.train.certified.util import synchronise_bn
from CTRAIN.util.util import preserve_model_state


def clean_bn_statistics(model, data, layers):
    """Detached clean statistics; PyTorch updates running buffers once per batch."""
    statistics = {}
    with torch.no_grad():
        for layer in layers:
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                dims = (0, *range(2, data.ndim))
                if layer.training:
                    statistics[layer] = (data.mean(dims).detach(), data.var(dims, unbiased=False).detach())
                else:
                    statistics[layer] = (layer.running_mean.detach().clone(), layer.running_var.detach().clone())
            data = layer(data)
    return statistics


def scheduled_epsilon(eps, epoch, batch, batches, warm_up_epochs, ramp_up_epochs):
    if epoch < warm_up_epochs:
        return 0.
    if ramp_up_epochs == 0:
        return eps
    return eps * min(1., (epoch - warm_up_epochs + (batch + 1) / batches) / ramp_up_epochs)


def set_based_train_model(model, train_loader, optimizer, num_epochs, eps,
                          val_loader=None, eval_eps=None, warm_up_epochs=1,
                          ramp_up_epochs=20, lr_scheduler=None, tau=.1,
                          input_set='box', num_generators=8, error_propagation='interval',
                          propagation_batch_size=1, pgd_eps_factor=1., population_bn=False,
                          gradient_clip=10., start_epoch=0, end_epoch=None,
                          results_path=None, checkpoint_save_interval=10, device='cpu',
                          selection_state=None):
    validate_set_options(tau, input_set, num_generators, error_propagation, propagation_batch_size, pgd_eps_factor)
    if not math.isfinite(eps) or eps < 0 or (eval_eps is not None and (not math.isfinite(eval_eps) or eval_eps < 0)):
        raise ValueError('Training and evaluation radii must be finite and nonnegative')
    if type(num_epochs) is not int or num_epochs <= 0:
        raise ValueError('num_epochs must be a positive integer')
    for name, value in [('warm_up_epochs', warm_up_epochs), ('ramp_up_epochs', ramp_up_epochs), ('start_epoch', start_epoch)]:
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    end_epoch = num_epochs if end_epoch is None else end_epoch
    if type(end_epoch) is not int or not start_epoch < end_epoch <= num_epochs:
        raise ValueError('Require start_epoch < end_epoch <= num_epochs')
    if type(checkpoint_save_interval) is not int or checkpoint_save_interval <= 0:
        raise ValueError('checkpoint_save_interval must be a positive integer')
    if gradient_clip is not None and (not math.isfinite(gradient_clip) or gradient_clip <= 0):
        raise ValueError('gradient_clip must be positive and finite or None')
    layers = sequential_layers(model)
    if not len(train_loader):
        raise ValueError('Training loader must contain batches')
    selection = selection_state if selection_state is not None else {}
    best_accuracy = selection.get('accuracy', -1.)
    best_weights = selection.get('weights')
    output_path = None if results_path is None else Path(results_path)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    with preserve_model_state(model):
        model.train()
        for epoch in range(start_epoch, end_epoch):
            total_loss = samples = 0
            epoch_start = last_update = time.monotonic()
            with progress_bar(train_loader, epoch, num_epochs, scheduled_epsilon(eps, epoch, 0, len(train_loader), warm_up_epochs, ramp_up_epochs), method="SetBased") as progress:
                for batch, (data, target) in enumerate(progress):
                    data, target = data.to(device), target.to(device)
                    current_eps = scheduled_epsilon(eps, epoch, batch, len(train_loader), warm_up_epochs, ramp_up_epochs)
                    radius = loader_radius(current_eps, train_loader, data)
                    limits = loader_limits(train_loader, data)
                    statistics = clean_bn_statistics(model, data, layers)
                    # Construct attacks once per logical batch so RNG use is independent of chunk size.
                    attack = None
                    if current_eps > 0 and input_set == 'attack':
                        attack = attack_zonotope(model, data, target, radius, *limits, num_generators, pgd_eps_factor)
                    optimizer.zero_grad(set_to_none=True)
                    for start in range(0, len(data), propagation_batch_size):
                        chunk = data[start:start + propagation_batch_size]
                        subset = None if attack is None else Zonotope(
                            attack.c[start:start + len(chunk)], attack.G[start:start + len(chunk)], attack.r[start:start + len(chunk)])
                        loss = get_set_based_loss(model, chunk, target[start:start + len(chunk)], current_eps,
                                                  radius, *limits, tau=tau, input_set=input_set,
                                                  num_generators=num_generators, error_propagation=error_propagation,
                                                  pgd_eps_factor=pgd_eps_factor, bn_statistics=statistics,
                                                  layers=layers, input_zonotope=subset)
                        (loss * (len(chunk) / len(data))).backward()
                        total_loss += loss.item() * len(chunk)
                        if time.monotonic() - last_update >= 2:
                            progress.set_postfix(eps=f"{current_eps:.5g}", chunk=f"{start + len(chunk)}/{len(data)}", loss=f"{loss.item():.4f}")
                            last_update = time.monotonic()
                    if gradient_clip is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()
                    samples += len(data)
                    update_progress(progress, total_loss / samples, lr=optimizer.param_groups[0]["lr"])
            progress.write(f"SetBased Epoch {epoch + 1}/{num_epochs}: loss={total_loss / samples:.4f}, eps={current_eps:.5g}, seconds={time.monotonic() - epoch_start:.1f}")
            if population_bn:
                synchronise_bn(model, model, train_loader, device, population=True)
            if lr_scheduler is not None:
                lr_scheduler.step()
            if val_loader is not None and epoch + 1 >= warm_up_epochs + ramp_up_epochs:
                progress.write("Evaluating full-box zonotope accuracy...")
                _, accuracy = eval_zonotope(model, val_loader, eps if eval_eps is None else eval_eps,
                                             propagation_batch_size=propagation_batch_size,
                                             error_propagation=error_propagation, device=device)
                if accuracy > best_accuracy:
                    best_accuracy, best_weights = accuracy, copy.deepcopy(model.state_dict())
                    selection.update(accuracy=accuracy, weights=best_weights, epoch=epoch + 1)
            if output_path is not None and ((epoch + 1) % checkpoint_save_interval == 0 or epoch + 1 == end_epoch):
                # Save the trajectory BEFORE inference-only best-weight restoration.
                torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                                lr_scheduler_state_dict=None if lr_scheduler is None else lr_scheduler.state_dict(),
                                epoch=epoch + 1, loss=total_loss / samples, selection_state=selection),
                           output_path / f'checkpoint_{epoch + 1}.pt')
        if val_loader is not None and best_weights is not None:
            model.load_state_dict(best_weights)
    return model
