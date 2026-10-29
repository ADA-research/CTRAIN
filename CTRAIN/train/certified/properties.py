"""Train logical output properties using bounds over a covering set of boxes.

Task examples and property regions are independent: examples train predictions,
while a selected bound or adversarial loss trains the formula on sampled regions.
Wrappers supply explicit configuration and a PropertyTrainingState trajectory.
"""
from dataclasses import dataclass, field
from pathlib import Path
import random
import numpy as np
import json
import math
import torch
from torch import nn
from CTRAIN.eval.properties import (
    evaluate_cells, refine_partition, evaluate_property_dataset, partition_state, validate_partition,
)
from CTRAIN.train.certified.losses.property_methods import compute_property_training_loss
from CTRAIN.properties.training import resolve_training_options
from CTRAIN.train.certified.util import (
    validate_training_run, checkpoint_due, EpochMetrics,
    update_model_selection, restore_model_selection,
)
from CTRAIN.util import save_checkpoint
from CTRAIN.train.certified.eps_scheduler import linear_schedule_fraction
from CTRAIN.util.util import preserve_model_state
from CTRAIN.train.certified.progress import progress_bar, tqdm


@dataclass
class PropertyTrainingState:
    """Retain the state required to continue a property-training trajectory.

    The trainer updates the covering partition, completed epoch count, history
    and validation selection in place. A wrapper can expose these fields without
    owning the numerical training procedure.

    Attributes:
        partition (list[dict]): Covering cells, split lineages and sampling probabilities.
        epoch (int): Number of completed epochs.
        history (list[dict]): Per-epoch task losses, property losses and cover diagnostics.
        selection_state (dict): Best validation score, inference weights and selected epoch.
        inference_selected (bool): Whether selected inference weights have replaced trajectory weights.
        loader_rng (object): Pending loader-generator state for checkpoint resumption.
    """
    partition: list
    epoch: int = 0
    history: list = field(default_factory=list)
    selection_state: dict = field(default_factory=dict)
    inference_selected: bool = False
    loader_rng: object = None


def save_property_checkpoint(path, original_model, optimizer, lr_scheduler, state,
                             configuration, train_loader=None):
    """Save a property-training trajectory, including its cover, configuration and RNG states."""
    if state.inference_selected:
        raise ValueError('Selected inference weights are not a resumable trajectory; load an epoch checkpoint first')
    path = Path(path)
    extra = dict(configuration=configuration, partition=partition_state(state.partition),
        history=state.history,
        rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        loader_rng=(train_loader.generator.get_state() if train_loader is not None
                    and train_loader.generator is not None else None))
    loss = state.history[-1]['loss'] if state.history and 'loss' in state.history[-1] else None
    save_checkpoint(original_model, optimizer, loss, state.epoch, path.parent,
                    lr_scheduler=lr_scheduler, selection_state=state.selection_state,
                    extra_state=extra, filename=path.name)


def _recalibrate(original_model, bounds, loader, reference, population_bn):
    """Refresh property-training BatchNorm statistics and synchronise verification buffers when
    requested.

    Only BatchNorm layers enter training mode; all other layers stay in eval
    mode. Batch statistics are averaged equally across loader batches. Caller
    modes and BN momentum are restored, including if recalibration fails.
    """
    layers = [m for m in original_model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not population_bn or not layers:
        bounds.sync_buffers()
        return
    momenta = [m.momentum for m in layers]
    try:
        with preserve_model_state(original_model), torch.no_grad():
            original_model.eval()
            for m in layers:
                m.train()
            for index, (data, _) in enumerate(loader, 1):
                for m in layers:
                    m.momentum = 1 / index
                original_model(data.to(reference))
    finally:
        for m, momentum in zip(layers, momenta):
            m.momentum = momentum
        bounds.sync_buffers()


def _task_value(original_model, task_loss, loader, reference):
    """Compute the sample-weighted task loss without accumulating gradients."""
    total, count = 0., 0
    with torch.no_grad():
        for data, target in loader:
            target = target.to(reference) if target.is_floating_point() else target.to(reference.device)
            loss = task_loss(original_model(data.to(reference)), target).mean()
            total += float(loss) * len(data)
            count += len(data)
    if not count:
        raise ValueError('Validation loader must not be empty')
    return total / count


def property_train_model(original_model=None, train_loader=None, val_loader=None,
                         start_epoch=None, end_epoch=None, *, bounds=None,
                         verification_model=None, properties=None, task_loss=None,
                         optimizer=None, lr_scheduler=None, num_epochs=None, state=None,
                         training_method='CROWN-IBP', training_options=None,
                         property_weight=1., margin=1e-3, warm_up_epochs=0,
                         weight_ramp_epochs=0, property_batch_size=32,
                         match_property_batch_size=False, refinement=False,
                         split_count=8, max_cells=1024, max_depth=12,
                         population_bn=False, property_dataset=None,
                         results_path=None, checkpoint_save_interval=10,
                         configuration=None, epoch_metrics=None, wrapper=None):
    """Optimise task and property losses, then freshly bound the whole cover.

    For each task batch, sample covering cells with replacement according to
    their stored weights. A lazy property dataset instead samples uniformly from
    all its boxes with replacement. Group cells by specification and compute
    lower margins using the selected training strategy. IBP/CROWN use full regions;
    MTL/CC/Exp combine adversarial and IBP components; SABR bounds contained
    subregions; PGD uses adversarial points; IBP-R adds unstable ReLU areas;
    GLS uses its existing Gaussian optimiser with a fixed sampled cover per
    update. Nested disjunctions retain their compiled graph.
    The update minimises::

        mean(task_loss) + property_weight * ramp * mean(sampled_cell_penalties)

    The default hinge penalty is `sum(max(0, margin - training_margin))`;
    cross-entropy uses the padded-margin loss. Training margins may be bounds
    or adversarial point margins, depending on the selected strategy.
    The first `warm_up_epochs` optimise only the task loss, without sampling or
    evaluating properties. The ramp starts afterward and reaches one after
    `weight_ramp_epochs`; zero ramp epochs apply the full weight after warm-up.
    Both phases are included in `num_epochs`. Cells describe whole input regions;
    the strategy chooses its training computation within them. Splitting preserves a parent's total
    sampling weight, so refinement does not overweight difficult regions.

    After each epoch: step the scheduler, optionally recalibrate BatchNorm,
    evaluate every covering cell with the training method's full-box bound
    (IBP for hybrids/adversarial training), optionally split uncertified cells and
    reevaluate, then record history and select/checkpoint weights. Certification
    requires every obligation on every cell to have a finite positive lower
    margin. A negative bound means unresolved, not a proven counterexample.

    All graphs run in evaluation mode with weight gradients enabled, keeping
    BatchNorm statistics fixed during optimiser updates and disabling dropout.
    Caller modes and parameter gradient flags are restored even on failure.

    Shared lifecycle arguments have the same meaning as in classifier trainers.
    ``state`` holds the partition, completed epoch, history, selection and loader
    RNG state; the trainer updates it in place. Models, bounds, losses and method
    options are explicit computational inputs. ``configuration`` is required for
    checkpointing so incompatible resumes can be rejected.

    The legacy ``property_train_model(wrapper, loader, ...)`` entry point remains
    supported, including the ``wrapper=`` keyword. New integrations should pass
    the task model and explicit inputs, as PropertyModelWrapper does.

    Returns the task model, restoring inference-selected weights only when a
    validation loader is supplied. Checkpoints precede that restoration.
    """
    if wrapper is not None:
        if original_model is not None:
            raise ValueError('Pass original_model or legacy wrapper, not both')
        original_model = wrapper
    if bounds is None and hasattr(original_model, 'bounds'):
        return original_model.train_model(train_loader, val_loader, start_epoch, end_epoch,
                                          epoch_metrics=epoch_metrics)
    if (bounds is None or optimizer is None or not callable(task_loss)
            or not isinstance(state, PropertyTrainingState)):
        raise ValueError('bounds, optimizer, task_loss and PropertyTrainingState are required')
    verification_model = bounds.model if verification_model is None else verification_model
    properties = bounds.specs if properties is None else properties
    reference = next(original_model.parameters(), None)
    if reference is None or not reference.is_floating_point():
        raise ValueError('Task model requires floating-point trainable parameters')
    if bounds.model is not verification_model or not any(m is original_model for m in verification_model.modules()):
        raise ValueError('Bounds and verification_model must share the supplied task model')
    training_method, training_options = resolve_training_options(training_method, training_options)
    if training_method == 'GLS':
        from CTRAIN.train.certified.gls import GLSOptimizer
        if not isinstance(optimizer, GLSOptimizer):
            raise ValueError('GLS property training requires GLSOptimizer')
    selection_bound_method = (training_method if training_method in ('IBP', 'CROWN', 'CROWN-IBP')
                              else training_options.get('bound_method', 'IBP'))
    if results_path is not None and configuration is None:
        raise ValueError('Checkpointing requires the property/training configuration')
    for name, value in dict(warm_up_epochs=warm_up_epochs, weight_ramp_epochs=weight_ramp_epochs,
                            max_depth=max_depth).items():
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    for name, value in dict(property_batch_size=property_batch_size, split_count=split_count,
                            max_cells=max_cells).items():
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not math.isfinite(property_weight) or property_weight < 0 or not math.isfinite(margin) or margin <= 0:
        raise ValueError('property_weight must be nonnegative finite and margin positive finite')
    if epoch_metrics is not None and not callable(epoch_metrics):
        raise ValueError('epoch_metrics must be callable')
    if state.inference_selected:
        raise ValueError('Load an epoch checkpoint before resuming a validation-selected model')
    start = state.epoch if start_epoch is None else start_epoch
    end = num_epochs if end_epoch is None else end_epoch
    end = validate_training_run(train_loader, num_epochs, start, end, checkpoint_save_interval)
    if warm_up_epochs + weight_ramp_epochs > num_epochs:
        raise ValueError('Warm-up and weight ramp must fit num_epochs')
    if val_loader is not None and not len(val_loader):
        raise ValueError('Validation loader must not be empty')
    if property_dataset is None:
        validate_partition(state.partition, properties)
    elif len(properties) != 1 or refinement or not len(property_dataset) or state.partition:
        raise ValueError('A lazy property dataset requires one formula, nonempty boxes and no refinement/partition')
    output_path = None if results_path is None else Path(results_path)
    if start != state.epoch:
        raise ValueError('Resume with load_checkpoint rather than skipping optimizer/partition state')
    if state.loader_rng is not None:
        if train_loader.generator is None:
            raise ValueError('Resumed checkpoint requires a loader with its own generator')
        train_loader.generator.set_state(state.loader_rng)
        state.loader_rng = None
    models = (original_model, verification_model, *bounds.modules)
    with preserve_model_state(*models):
        for model in models:
            model.eval()  # fixed-statistics, deterministic property training
        for epoch in range(start, end):
            metrics = EpochMetrics()
            with progress_bar(train_loader, epoch, num_epochs, None,
                              method=f'Properties {training_method}') as progress:
                for batch_index, (data, target) in enumerate(progress):
                    optimizer.zero_grad(set_to_none=True)
                    fraction = linear_schedule_fraction(epoch, batch_index, len(train_loader),
                                                        warm_up_epochs, weight_ramp_epochs,
                                                        at_batch_end=True)
                    property_active = property_weight > 0 and fraction > 0
                    sample_count = len(data) if match_property_batch_size else property_batch_size
                    batch_boxes, sampled = None, []
                    if property_active and property_dataset is None:
                        weights = torch.tensor([c['probability'] for c in state.partition])
                        sampled = torch.multinomial(weights, sample_count, replacement=True).tolist()
                        batch_boxes = None
                    elif property_active:
                        # Uniform sampling with replacement from the entire lazy
                        # box pool, independently of the current task batch.
                        indices = torch.randint(len(property_dataset),
                                                (sample_count,)).tolist()
                        batch_boxes = [property_dataset[i] for i in indices]
                    def closure():
                        output = original_model(data.to(reference))
                        labels = target.to(reference) if target.is_floating_point() else target.to(reference.device)
                        task = task_loss(output, labels).mean()
                        property_term = task.new_zeros(())
                        if batch_boxes is not None:
                            property_term = compute_property_training_loss(bounds, 0, batch_boxes,
                                verification_model=verification_model, training_method=training_method,
                                training_options=training_options, margin=margin, reference=reference)
                        for p in (range(len(properties)) if property_active and batch_boxes is None else ()):
                            cells = [state.partition[i] for i in sampled if state.partition[i]['property'] == p]
                            if cells:
                                term = compute_property_training_loss(bounds, p, [c['box'] for c in cells],
                                    verification_model=verification_model, training_method=training_method,
                                    training_options=training_options, margin=margin, reference=reference)
                                property_term = property_term + term * len(cells) / len(sampled)
                        loss = task + property_weight * fraction * property_term
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Non-finite task/property training loss')
                        return loss, task, property_term
                    if training_method == 'GLS':
                        optimizer.step(lambda: closure()[0])
                        # Record centre-model components after the Gaussian update.
                        with torch.no_grad():
                            loss, task, property_term = closure()
                    else:
                        loss, task, property_term = closure()
                        loss.backward()
                        if any(p.grad is not None and not torch.isfinite(p.grad).all()
                               for p in original_model.parameters()):
                            raise FloatingPointError('Non-finite property-training gradients')
                        clip = training_options['gradient_clip']
                        if clip is not None:
                            nn.utils.clip_grad_norm_(original_model.parameters(), clip)
                        optimizer.step()
                    metrics.update(len(data), loss=loss.detach(), task_loss=task.detach(),
                                   property_loss=property_term.detach())
                    progress.set_postfix(task=f'{metrics.mean("task_loss"):.4g}',
                                         property=f'{metrics.mean("property_loss"):.4g}', refresh=False)
            _recalibrate(original_model, bounds, train_loader, reference, population_bn)
            if lr_scheduler is not None:
                lr_scheduler.step()
            options = training_options
            if training_method == 'GLS' and epoch + 1 in options['sigma_decay_milestones']:
                with torch.no_grad():
                    for scale in optimizer.scales:
                        scale.mul_(options['sigma_decay_factor'])
            if epoch < warm_up_epochs:
                # Warm-up is task-only, including diagnostics and refinement.
                minimum, certified = -float('inf'), None
                certified_count = None
                cell_count = len(property_dataset) if property_dataset is not None else len(state.partition)
            elif property_dataset is not None:
                coverage = evaluate_property_dataset(bounds, property_dataset, selection_bound_method, property_batch_size)
                minimum = coverage['minimum_lower_bound']
                minimum = minimum if minimum is not None else -float('inf')
                certified, cell_count = coverage['certified'], coverage['total_boxes']
                certified_count = coverage['certified_count']
            else:
                rows, _ = evaluate_cells(bounds, state.partition, selection_bound_method,
                                         property_batch_size)
                if refinement:
                    state.partition = refine_partition(state.partition, rows, properties,
                        split_count, max_cells, max_depth)
                    rows, _ = evaluate_cells(bounds, state.partition, selection_bound_method,
                                             property_batch_size)
                minimum = min((r['minimum_lower_bound'] if r['minimum_lower_bound'] is not None
                               else -float('inf')) for r in rows)
                certified, cell_count = all(r['outcome'] == 'certified' for r in rows), len(rows)
                certified_count = sum(r['outcome'] == 'certified' for r in rows)
            valid = _task_value(original_model, task_loss, val_loader, reference) if val_loader is not None else None
            if (val_loader is not None and epoch >= warm_up_epochs
                    and epoch + 1 >= warm_up_epochs + weight_ramp_epochs):
                # Prefer a complete certificate over any partial-cover score.
                score = (int(certified), -valid if certified else minimum, -valid)
                update_model_selection(state.selection_state, original_model, score, epoch + 1,
                                       allow_negative_infinity=True)
            state.epoch = epoch + 1
            entry = dict(epoch=state.epoch, loss=metrics.mean("loss"),
                         task_loss=metrics.mean("task_loss"), property_loss=metrics.mean("property_loss"),
                         validation_loss=valid,
                         phase=('warm_up' if epoch < warm_up_epochs else
                                'weight_ramp' if epoch < warm_up_epochs + weight_ramp_epochs
                                else 'full_weight'),
                         effective_property_weight=property_weight * fraction,
                         property_evaluated=epoch >= warm_up_epochs,
                         certified=certified, cells=cell_count,
                         certified_count=certified_count,
                         certified_fraction=None if certified_count is None else certified_count / cell_count,
                         minimum_lower_bound=minimum if math.isfinite(minimum) else None)
            if epoch_metrics is not None:
                with torch.no_grad(), preserve_model_state(original_model):
                    original_model.eval()
                    entry['metrics'] = epoch_metrics(original_model)
                if not isinstance(entry['metrics'], dict):
                    raise ValueError('epoch_metrics must return a metrics dict')
                json.dumps(entry['metrics'], allow_nan=False)
            state.history.append(entry)
            certificate = ('not evaluated (warm-up)' if certified_count is None else
                           f'{certified_count}/{cell_count} ({entry["certified_fraction"]:.1%})')
            diagnostics = ' ' + json.dumps(entry['metrics'], allow_nan=False) if 'metrics' in entry else ''
            tqdm.write(f'Property epoch {state.epoch}/{num_epochs}: task={entry["task_loss"]:.6g} '
                       f'property={entry["property_loss"]:.6g} certified={certificate}{diagnostics}')
            if output_path is not None:
                output_path.mkdir(parents=True, exist_ok=True)
                history_path = output_path / 'history.json'
                temporary = history_path.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(state.history, indent=2, allow_nan=False))
                temporary.replace(history_path)
            if output_path is not None and (
                    checkpoint_due(state.epoch, end, checkpoint_save_interval)):
                save_property_checkpoint(output_path / f'{state.epoch}_checkpoint.pt', original_model,
                    optimizer, lr_scheduler, state, configuration, train_loader)
        if val_loader is not None and restore_model_selection(state.selection_state, original_model):
            bounds.sync_buffers()
            state.inference_selected = True
    return original_model
