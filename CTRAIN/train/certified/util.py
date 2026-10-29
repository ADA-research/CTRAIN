import copy
import math
import torch
import numpy as np


def validate_training_run(train_loader, num_epochs, start_epoch, end_epoch,
                          checkpoint_save_interval, lr_decay_schedule_unit='epoch', *, history=None):
    """Validate the shared lifecycle before initialization or optimizer updates."""
    end_epoch = num_epochs if end_epoch is None else end_epoch
    if (any(type(v) is not int for v in (num_epochs, start_epoch, end_epoch))
            or num_epochs <= 0 or not 0 <= start_epoch <= end_epoch <= num_epochs):
        raise ValueError('Require 0 <= start_epoch <= end_epoch <= num_epochs, with positive num_epochs')
    if type(checkpoint_save_interval) is not int or checkpoint_save_interval <= 0:
        raise ValueError('checkpoint_save_interval must be a positive integer')
    if not len(train_loader):
        raise ValueError('Training loader must contain batches')
    if lr_decay_schedule_unit not in ('epoch', 'batch'):
        raise ValueError("lr_decay_schedule_unit must be 'epoch' or 'batch'")
    if history is not None and not isinstance(history, list):
        raise ValueError('history must be a list')
    return end_epoch


def resolve_validation_eps(validation_eps, eval_eps):
    """Accept the legacy evaluation-radius name without silently overriding it."""
    if validation_eps is not None and eval_eps is not None and validation_eps != eval_eps:
        raise ValueError('validation_eps and legacy eval_eps must agree')
    return eval_eps if validation_eps is None else validation_eps


def checkpoint_due(epoch, end_epoch, interval):
    """Save periodic and final trajectory states before inference selection."""
    return epoch % interval == 0 or epoch == end_epoch


class EpochMetrics:
    """Accumulate batch means by example count, including partial batches."""
    def __init__(self):
        self.samples = 0
        self.totals = {}

    def update(self, count, **values):
        self.samples += count
        for key, value in values.items():
            self.totals[key] = self.totals.get(key, 0.) + float(value) * count

    def mean(self, key):
        return self.totals[key] / self.samples


def evaluate_taps_proxy(original_model, hardened_model, data_loader, eps, n_classes,
                        device, propagation="IBP", sabr_args=None, **attack_args):
    """Evaluate latent-margin accuracy without updating BatchNorm statistics."""
    from auto_LiRPA import PerturbationLpNorm
    from CTRAIN.bound import bound_taps
    from CTRAIN.util.util import preserve_model_state

    models = (original_model, hardened_model, *hardened_model.bounded_blocks)
    correct = total = 0
    with preserve_model_state(*models), torch.no_grad():
        for model in models:
            model.eval()
        radius = torch.as_tensor(eps, device=device).reshape(-1, 1, 1)
        data_min, data_max = data_loader.min.to(device), data_loader.max.to(device)
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            if (radius == 0).all():
                correct += (hardened_model(data).argmax(1) == target).sum().item()
            else:
                ptb = PerturbationLpNorm(
                    norm=np.inf, eps=radius,
                    x_L=torch.clamp(data - radius, data_min, data_max),
                    x_U=torch.clamp(data + radius, data_min, data_max),
                )
                region_args = None if sabr_args is None else dict(
                    sabr_args, hardened_model=hardened_model, original_model=original_model,
                    data=data, target=target, eps=radius, data_min=data_min,
                    data_max=data_max, device=device, n_classes=n_classes,
                )
                margins, _ = bound_taps(
                    original_model, hardened_model, hardened_model.bounded_blocks,
                    data, target, n_classes, ptb, device=device,
                    propagation=propagation, sabr_args=region_args, **attack_args,
                )
                correct += (margins > 0).all(1).sum().item()
            total += len(data)
    if not total:
        raise ValueError("Validation loader must contain at least one sample")
    return correct / total


def synchronise_bn(original_model, hardened_model, train_loader, device, population=False, loss_fusion_model=None):
    """Export current full-model buffers; optionally use CTBench population BN."""
    import torch
    from torch.nn.modules.batchnorm import _BatchNorm
    from CTRAIN.util.util import preserve_model_state

    original = original_model
    with torch.no_grad():
        buffers = dict(original.named_buffers())
        for name, value in hardened_model.named_buffers():
            buffers[name].copy_(value)
        layers = [m for m in original.modules() if isinstance(m, _BatchNorm)]
        if population and layers:
            momenta = [m.momentum for m in layers]
            try:
                with preserve_model_state(original):
                    original.train()
                    for batch, (data, _) in enumerate(train_loader, 1):
                        for layer in layers:
                            layer.momentum = 1 / batch
                        original(data.to(device))
            finally:
                for layer, momentum in zip(layers, momenta):
                    layer.momentum = momentum
        buffers = dict(original.named_buffers())
        for name, value in hardened_model.named_buffers():
            value.copy_(buffers[name])
        if loss_fusion_model is not None:
            buffers = dict(original.named_buffers())
            for name, value in loss_fusion_model.named_buffers():
                value.copy_(buffers[name.removeprefix("model.")])


def prepare_model_selection(state, loader, eps=None):
    if state is not None and not isinstance(state, dict):
        raise ValueError('selection_state must be a dictionary')
    if loader is not None and not len(loader):
        raise ValueError('Validation loader must contain at least one sample')
    if eps is not None and (not math.isfinite(eps) or eps < 0):
        raise ValueError('Validation radius must be nonnegative and finite')
    return {} if state is None else state


def update_model_selection(state, model, score, epoch, *, allow_negative_infinity=False):
    """Keep the earliest best inference weights, separate from the trajectory."""
    score = tuple(float(v) for v in score) if isinstance(score, tuple) else float(score)
    values = score if isinstance(score, tuple) else (score,)
    if any(not math.isfinite(v) and not (allow_negative_infinity and v == -float('inf')) for v in values):
        raise FloatingPointError('Validation score must be finite')
    previous = state.get('score', state.get('accuracy'))
    if previous is None or score > previous:
        state.update(score=score, weights=copy.deepcopy(model.state_dict()), epoch=epoch)


def restore_model_selection(state, model):
    if state.get('weights') is None:
        return False
    model.load_state_dict(state['weights'])
    return True


def evaluate_validation_loss(models, loader, loss_func, device):
    """Sample-weighted robust data loss with fixed evaluation-mode BN statistics."""
    from CTRAIN.util.util import preserve_model_state
    total = samples = 0
    with preserve_model_state(*models), torch.no_grad():
        for model in models:
            model.eval()
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            loss = loss_func(data, target)
            total += float(loss) * len(data)
            samples += len(data)
    if not samples:
        raise ValueError('Validation loader must contain at least one sample')
    return total / samples


def validation_ptb(data, loader, eps):
    from auto_LiRPA import PerturbationLpNorm
    from CTRAIN.util.util import loader_radius, loader_limits
    radius = loader_radius(eps, loader, data)
    minimum, maximum = loader_limits(loader, data)
    return PerturbationLpNorm(norm=float('inf'), eps=radius,
                             x_L=(data - radius).clamp(minimum, maximum),
                             x_U=(data + radius).clamp(minimum, maximum))
