"""Full clipped-box zonotope certification, independent of training input sets."""
import copy
import math
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from CTRAIN.util import construct_c
from CTRAIN.bound.zonotope import box_zonotope, bound_zonotope, sequential_layers, zonotope_margins
from CTRAIN.util.util import preserve_model_state


class _BoxPerturbation(PerturbationLpNorm):
    def init_sparse_linf(self, x, x_L, x_U):
        # Installed LiRPA builds sparse coefficients in the default dtype.
        # Keep its sparse construction, correcting only float64 compatibility.
        linear, center, aux = super().init_sparse_linf(x, x_L, x_U)
        linear.lw, linear.uw = linear.lw.to(x), linear.uw.to(x)
        return linear, center, aux


def loader_radius(eps, loader, data):
    radius = torch.as_tensor(eps, device=data.device, dtype=data.dtype)
    if getattr(loader, 'normalised', False):
        radius = radius / torch.as_tensor(loader.std, device=data.device, dtype=data.dtype)
    if radius.ndim == 1 and data.ndim > 2:
        radius = radius.reshape(-1, *([1] * (data.ndim - 2)))
    return radius


def loader_limits(loader, data):
    return tuple(torch.as_tensor(getattr(loader, name), device=data.device, dtype=data.dtype)
                 for name in ('min', 'max'))


def eval_zonotope(model, data_loader, eps, test_samples=float('inf'),
                  propagation_batch_size=1, error_propagation='interval', device=None):
    """Return clean and certified accuracy fractions. Always use fresh full boxes."""
    if type(propagation_batch_size) is not int or propagation_batch_size <= 0:
        raise ValueError('propagation_batch_size must be a positive integer')
    if test_samples <= 0:
        raise ValueError('test_samples must be positive')
    if not math.isfinite(eps) or eps < 0:
        raise ValueError('eps must be nonnegative and finite')
    if error_propagation not in ('interval', 'generators'):
        raise ValueError('error_propagation must be interval or generators')
    layers = sequential_layers(model)
    bounded = None
    device = next(model.parameters()).device if device is None else device
    clean = certified = total = 0
    with preserve_model_state(model), torch.no_grad():
        model.eval()
        for data, target in data_loader:
            count = int(min(len(data), test_samples - total))
            if count <= 0:
                break
            data, target = data[:count].to(device), target[:count].to(device)
            radius = loader_radius(eps, data_loader, data)
            lower_limit, upper_limit = loader_limits(data_loader, data)
            logits = model(data)
            clean += (logits.argmax(1) == target).sum().item()
            for start in range(0, count, propagation_batch_size):
                chunk = data[start:start + propagation_batch_size]
                labels = target[start:start + len(chunk)]
                lower = (chunk - radius).clamp(lower_limit, upper_limit)
                upper = (chunk + radius).clamp(lower_limit, upper_limit)
                if error_propagation == 'interval':
                    if bounded is None:
                        # Conversion may touch BN buffers; isolate the authoritative model.
                        bounded = BoundedModule(copy.deepcopy(model), chunk,
                                                bound_opts={'activation_bound_option': 'same-slope'},
                                                device=device)
                        bounded.eval()
                    inputs = BoundedTensor(chunk, _BoxPerturbation(norm=float('inf'),
                                                                    x_L=lower, x_U=upper))
                    margins, _ = bounded.compute_bounds(
                        x=(inputs,), C=construct_c(chunk, labels, logits.shape[1]),
                        method='Forward', bound_upper=False)
                else:
                    # Dynamic forward lacks BN support in the installed LiRPA;
                    # retain full error-generator semantics for this optional mode.
                    output = bound_zonotope(model, box_zonotope(lower, upper),
                                            error_propagation, layers=layers)
                    margins = zonotope_margins(output, labels)
                certified += (margins > 0).all(1).sum().item()
            total += count
    if not total:
        raise ValueError('Evaluation loader must contain samples')
    return clean / total, certified / total


def evaluate_zonotope_model(model, loader, eps, test_samples=float('inf'),
                             propagation_batch_size=1, error_propagation='interval', device=None):
    """Match CTRAIN's (clean, certified, adversarial) evaluation interface."""
    from CTRAIN.eval.eval import eval_adversarial
    clean, certified = eval_zonotope(model, loader, eps, test_samples,
                                     propagation_batch_size, error_propagation, device)
    device = next(model.parameters()).device if device is None else device
    with preserve_model_state(model):
        data = next(iter(loader))[0].to(device)
        radius = loader_radius(eps, loader, data)
        model.eval()
        n_classes = model(data[:1]).shape[1]
        adversarial = eval_adversarial(model, loader, radius, n_classes=n_classes,
                                       device=device, test_samples=test_samples)
    return clean, certified, adversarial
