"""Train logical output properties using bounds over a covering set of boxes.

Task examples and property regions are independent: examples train predictions,
while a selected bound or adversarial loss trains the formula on sampled regions.
`PropertyModelWrapper` supplies configuration and owns all resumable state.
"""
import copy
import math
import torch
from torch import nn
from CTRAIN.eval.properties import evaluate_cells, refine_partition
from CTRAIN.train.certified.losses.property_methods import get_property_training_loss
from CTRAIN.util.util import preserve_model_state


def _recalibrate(wrapper, loader):
    """Optionally refresh BN running statistics and synchronize bound buffers.

    Only BatchNorm layers enter training mode; all other layers stay in eval
    mode. Batch statistics are averaged equally across loader batches. Caller
    modes and BN momentum are restored, including if recalibration fails.
    """
    layers = [m for m in wrapper.original_model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not wrapper.population_bn or not layers:
        wrapper.bounds.sync_buffers()
        return
    momenta = [m.momentum for m in layers]
    try:
        with preserve_model_state(wrapper.original_model), torch.no_grad():
            wrapper.original_model.eval()
            for m in layers:
                m.train()
            for index, (data, _) in enumerate(loader, 1):
                for m in layers:
                    m.momentum = 1 / index
                wrapper.original_model(data.to(wrapper.reference))
    finally:
        for m, momentum in zip(layers, momenta):
            m.momentum = momentum
        wrapper.bounds.sync_buffers()


def _task_value(wrapper, loader):
    """Average task loss by example count without accumulating gradients."""
    total, count = 0., 0
    with torch.no_grad():
        for data, target in loader:
            target = target.to(wrapper.reference) if target.is_floating_point() else target.to(wrapper.device)
            loss = wrapper.task_loss(wrapper.original_model(data.to(wrapper.reference)), target).mean()
            total += float(loss) * len(data)
            count += len(data)
    if not count:
        raise ValueError('Validation loader must not be empty')
    return total / count


def property_train_model(wrapper, train_loader, val_loader=None, start_epoch=None, end_epoch=None):
    """Optimize task and property losses, then freshly bound the whole cover.

    For each task batch, sample covering cells with replacement according to
    their stored weights. Group those cells by specification, compute lower
    margins using the selected training strategy. IBP/CROWN use full regions;
    MTL/CC/Exp combine adversarial and IBP components; SABR bounds contained
    subregions; PGD uses adversarial points; IBP-R adds unstable ReLU areas;
    GLS uses its existing Gaussian optimizer with a fixed sampled cover per
    update. Nested disjunctions retain their compiled graph.
    The update minimizes::

        mean(task_loss) + property_weight * ramp * mean(sampled_cell_penalties)

    The default hinge penalty is `sum(max(0, margin - training_margin))`;
    cross-entropy uses the padded-margin loss. Training margins may be bounds
    or adversarial point margins, depending on the selected strategy.
    The optional ramp reaches one after `weight_ramp_epochs`; zero ramp epochs
    apply the full weight immediately. Cells describe whole input regions;
    the strategy chooses its training computation within them. Splitting preserves a parent's total
    sampling weight, so refinement does not overweight difficult regions.

    After each epoch: step the scheduler, optionally recalibrate BatchNorm,
    evaluate every covering cell, optionally split uncertified cells and
    reevaluate, then record history and select/checkpoint weights. Certification
    requires every obligation on every cell to have a finite positive lower
    margin. A negative bound means unresolved, not a proven counterexample.

    All graphs run in evaluation mode with weight gradients enabled, keeping
    BatchNorm statistics fixed during optimizer updates and disabling dropout.
    Caller modes and parameter gradient flags are restored even on failure.

    Args:
        wrapper (PropertyModelWrapper): Configured model, bound graphs,
            optimizer/scheduler and trajectory state. Normally invoke this
            function through `wrapper.train_model(...)`.
        train_loader (torch.utils.data.DataLoader): Nonempty `(input, target)`
            task batches. Inputs/floating targets use the model device/dtype;
            integer target dtypes are preserved.
        val_loader (torch.utils.data.DataLoader | None): Optional nonempty
            loader for task loss. Selection prioritizes full certification,
            then lowest validation loss. Until certified, it prioritizes the
            highest minimum lower bound, with task loss breaking ties.
        start_epoch (int | None): Zero-based start, defaulting to `wrapper.epoch`.
            Must match that count; use `load_checkpoint` for resumption.
        end_epoch (int | None): Exclusive end, defaulting to `wrapper.num_epochs`.

    Returns:
        torch.nn.Module: The task model. With validation, restore selected
        inference weights and set `wrapper.inference_selected`; otherwise retain
        final weights. Selection does not restore optimizer/scheduler state.
        Automatic checkpoints save the trajectory before inference selection.

    Raises:
        ValueError: Epoch range or trajectory state is invalid, a loader is
            empty, or a saved explicit loader generator cannot be restored.
        FloatingPointError: Training loss or model gradients are non-finite.
    """
    if wrapper.inference_selected:
        raise ValueError('Load an epoch checkpoint before resuming a validation-selected model')
    start = wrapper.epoch if start_epoch is None else start_epoch
    end = wrapper.num_epochs if end_epoch is None else end_epoch
    if type(start) is not int or type(end) is not int or not 0 <= start <= end <= wrapper.num_epochs:
        raise ValueError('Invalid epoch range')
    if start != wrapper.epoch:
        raise ValueError('Resume with load_checkpoint rather than skipping optimizer/partition state')
    if not len(train_loader):
        raise ValueError('Training loader must not be empty')
    if wrapper.loader_rng is not None:
        if train_loader.generator is None:
            raise ValueError('Resumed checkpoint requires a loader with its own generator')
        train_loader.generator.set_state(wrapper.loader_rng)
        wrapper.loader_rng = None
    models = (wrapper.original_model, wrapper.verification_model, *wrapper.bounds.modules)
    with preserve_model_state(*models):
        for model in models:
            model.eval()  # fixed-statistics, deterministic property training
        for epoch in range(start, end):
            task_total, property_total, count = 0., 0., 0
            for batch_index, (data, target) in enumerate(train_loader):
                wrapper.optimizer.zero_grad(set_to_none=True)
                weights = torch.tensor([c['probability'] for c in wrapper.partition])
                # Regions are sampled independently of this task-data batch.
                sampled = torch.multinomial(weights, wrapper.property_batch_size, replacement=True).tolist()
                fraction = (min(1., (epoch + (batch_index + 1) / len(train_loader)) / wrapper.weight_ramp_epochs)
                            if wrapper.weight_ramp_epochs else 1.)
                def closure():
                    output = wrapper.original_model(data.to(wrapper.reference))
                    labels = target.to(wrapper.reference) if target.is_floating_point() else target.to(wrapper.device)
                    task = wrapper.task_loss(output, labels).mean()
                    property_term = task.new_zeros(())
                    for p in (range(len(wrapper.properties)) if wrapper.property_weight else ()):
                        cells = [wrapper.partition[i] for i in sampled if wrapper.partition[i]['property'] == p]
                        if cells:
                            term = get_property_training_loss(wrapper, p, [c['box'] for c in cells])
                            property_term = property_term + term * len(cells) / len(sampled)
                    loss = task + wrapper.property_weight * fraction * property_term
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite task/property training loss')
                    return loss, task, property_term
                if wrapper.training_method == 'GLS':
                    wrapper.optimizer.step(lambda: closure()[0])
                    # Record centre-model components after the Gaussian update.
                    with torch.no_grad():
                        _, task, property_term = closure()
                else:
                    loss, task, property_term = closure()
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all()
                           for p in wrapper.original_model.parameters()):
                        raise FloatingPointError('Non-finite property-training gradients')
                    clip = wrapper.training_options['gradient_clip']
                    if clip is not None:
                        nn.utils.clip_grad_norm_(wrapper.original_model.parameters(), clip)
                    wrapper.optimizer.step()
                task_total += float(task.detach()) * len(data)
                property_total += float(property_term.detach()) * len(data)
                count += len(data)
            if wrapper.lr_scheduler is not None:
                wrapper.lr_scheduler.step()
            options = wrapper.training_options
            if wrapper.training_method == 'GLS' and epoch + 1 in options['sigma_decay_milestones']:
                with torch.no_grad():
                    for scale in wrapper.optimizer.scales:
                        scale.mul_(options['sigma_decay_factor'])
            _recalibrate(wrapper, train_loader)
            rows, _ = evaluate_cells(wrapper.bounds, wrapper.partition, wrapper.bound_method,
                                     wrapper.property_batch_size)
            if wrapper.refinement:
                wrapper.partition = refine_partition(wrapper.partition, rows, wrapper.properties,
                    wrapper.split_count, wrapper.max_cells, wrapper.max_depth)
                rows, _ = evaluate_cells(wrapper.bounds, wrapper.partition, wrapper.bound_method,
                                         wrapper.property_batch_size)
            valid = _task_value(wrapper, val_loader) if val_loader is not None else None
            minimum = min((r['minimum_lower_bound'] if r['minimum_lower_bound'] is not None
                           else -float('inf')) for r in rows)
            certified = all(r['outcome'] == 'certified' for r in rows)
            if val_loader is not None:
                # Prefer a complete certificate over any partial-cover score.
                score = (int(certified), -valid if certified else minimum, -valid)
                if wrapper.best_score is None or score > wrapper.best_score:
                    wrapper.best_score, wrapper.best_weights = score, copy.deepcopy(wrapper.original_model.state_dict())
            wrapper.epoch = epoch + 1
            entry = dict(epoch=wrapper.epoch, task_loss=task_total / count,
                         property_loss=property_total / count, validation_loss=valid,
                         certified=certified, cells=len(rows),
                         minimum_lower_bound=minimum if math.isfinite(minimum) else None)
            wrapper.history.append(entry)
            print(f'Property epoch {wrapper.epoch}/{wrapper.num_epochs}: task={entry["task_loss"]:.6g} '
                  f'property={entry["property_loss"]:.6g} certified={certified} cells={len(rows)}', flush=True)
            if wrapper.checkpoint_path is not None and (
                    wrapper.epoch % wrapper.checkpoint_save_interval == 0 or wrapper.epoch == end):
                wrapper.save_checkpoint(wrapper.checkpoint_path / f'{wrapper.epoch}_checkpoint.pt', train_loader)
        if val_loader is not None and wrapper.best_weights is not None:
            wrapper.original_model.load_state_dict(wrapper.best_weights)
            wrapper.bounds.sync_buffers()
            wrapper.inference_selected = True
    return wrapper.original_model
