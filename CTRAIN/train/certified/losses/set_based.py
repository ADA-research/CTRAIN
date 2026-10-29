"""Set-based centre classification and enclosure-size objective."""
import math
import torch
from torch.nn import functional as F
from CTRAIN.bound.set_based_zono import Zonotope, box_zonotope, bound_zonotope
from CTRAIN.util.util import preserve_model_state


def validate_set_options(tau, input_set, num_generators, error_propagation,
                         propagation_batch_size, pgd_eps_factor):
    """Validate the enclosure-loss weight, input representation, propagation mode and attack options."""
    if not math.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError('tau must be finite and in [0, 1]')
    if input_set not in ('box', 'attack'):
        raise ValueError('input_set must be box or attack')
    for name, value in [('num_generators', num_generators), ('propagation_batch_size', propagation_batch_size)]:
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if error_propagation not in ('interval', 'generators'):
        raise ValueError('error_propagation must be interval or generators')
    if not math.isfinite(pgd_eps_factor) or pgd_eps_factor <= 0:
        raise ValueError('pgd_eps_factor must be positive and finite')


def attack_zonotope(model, data, target, radius, data_min, data_max,
                    num_generators=8, pgd_eps_factor=1., bn_statistics=None, layers=None):
    """Construct a contained input zonotope from random-start FGSM directions.

    Attack points are fitted coordinatewise to the nominal clipped input box.
    Scaling the search radius changes candidate generation, while the resulting
    zonotope remains within the nominal box. The construction returns detached
    centres and generators for subsequent differentiable propagation.
    """
    if type(num_generators) is not int or num_generators <= 0 or not math.isfinite(pgd_eps_factor) or pgd_eps_factor <= 0:
        raise ValueError('Invalid attack generator count or PGD epsilon factor')
    lower, upper = (data - radius).clamp(data_min, data_max), (data + radius).clamp(data_min, data_max)
    search_radius = radius * pgd_eps_factor
    search_lower = (data - search_radius).clamp(data_min, data_max)
    search_upper = (data + search_radius).clamp(data_min, data_max)
    positions = []
    # Attack candidates are independent of the nominal batch; retain its graph
    # for the later training loss, while holding these statistics fixed here.
    search_statistics = None if bn_statistics is None else {
        layer: (mean.detach(), variance.detach()) for layer, (mean, variance) in bn_statistics.items()}
    with preserve_model_state(model), torch.enable_grad():
        model.eval()
        # Candidates share the nominal statistics without updating buffers.
        groups = [num_generators] if bn_statistics is not None else [1] * num_generators
        for count in groups:
            point = torch.cat([search_lower + torch.rand_like(data) * (search_upper - search_lower)
                               for _ in range(count)]).detach().requires_grad_(True)
            if bn_statistics is None:
                logits = model(point)
            else:
                enclosure = Zonotope(point, point.new_empty((len(point), 0, *point.shape[1:])), torch.zeros_like(point))
                logits = bound_zonotope(model, enclosure, bn_statistics=search_statistics, layers=layers).c
            gradient, = torch.autograd.grad(F.cross_entropy(logits, target.repeat(count)), point)
            repeats = (count, *([1] * (data.ndim - 1)))
            step = (torch.zeros_like(data) + search_radius).repeat(repeats)
            point = (point + step * gradient.sign()).clamp(search_lower.repeat(repeats), search_upper.repeat(repeats))
            point = point.detach().clamp(lower.repeat(repeats), upper.repeat(repeats))
            positions.extend(point.reshape(count, *data.shape).unbind(0))
    positions = torch.stack(positions, 1)
    center = positions.mean(1).clamp(lower, upper)
    generators = (positions - center.unsqueeze(1)) / num_generators
    extent = generators.abs().sum(1)
    room = torch.minimum(center - lower, upper - center).clamp_min(0)
    divisor = torch.where(extent > 0, extent, torch.ones_like(extent))
    generators = generators * (room / divisor).clamp(max=1).unsqueeze(1)
    return Zonotope(center.detach(), generators.detach(), torch.zeros_like(center))


def get_set_based_loss(model, data, target, eps, radius, data_min, data_max,
                       tau=.1, input_set='box', num_generators=8,
                       error_propagation='interval', pgd_eps_factor=1.,
                       bn_statistics=None, layers=None, input_zonotope=None):
    """Combine centre classification with an output-enclosure size penalty.

    The objective weights centre cross-entropy by 1 - tau and the norm of the
    output generators and independent radii by tau. The enclosure term is
    normalised by the number of classes and the scheduled raw-input eps.
    At eps=0, the function returns ordinary centre cross-entropy.

    Args:
        model (torch.nn.Module): Supported sequential network.
        data (torch.Tensor): Nominal batch in model coordinates.
        target (torch.Tensor): Integer class labels.
        eps (float): Perturbation radius in model input coordinates.
        radius (torch.Tensor): Scheduled radius in model coordinates.
        data_min (torch.Tensor): Lower input-domain limit in model coordinates.
        data_max (torch.Tensor): Upper input-domain limit in model coordinates.
        tau (float): Enclosure-penalty weight in [0, 1].
        input_set (str): Training input representation: 'box' or an attack-generated
            'zonotope'.
        num_generators (int): Number of attack directions.
        error_propagation (str): 'interval' or 'generators' for nonlinear errors.
        pgd_eps_factor (float): Positive attack-search scaling factor.
        bn_statistics (dict, optional): Clean batch statistics shared with propagation.
        layers (list, optional): Previously validated sequential layers.
        input_zonotope (Zonotope, optional): Explicit training enclosure overriding set
            construction at positive eps.

    Returns:
        (torch.Tensor): Scalar set-based training loss.
    """
    if not math.isfinite(eps) or eps < 0:
        raise ValueError('eps must be nonnegative and finite')
    validate_set_options(tau, input_set, num_generators, error_propagation, 1, pgd_eps_factor)
    if eps == 0:
        enclosure = Zonotope(data, data.new_empty((len(data), 0, *data.shape[1:])), torch.zeros_like(data))
    elif input_zonotope is not None:
        enclosure = input_zonotope
    elif input_set == 'box':
        enclosure = box_zonotope((data - radius).clamp(data_min, data_max),
                                 (data + radius).clamp(data_min, data_max))
    else:
        enclosure = attack_zonotope(model, data, target, radius, data_min, data_max,
                                    num_generators, pgd_eps_factor, bn_statistics, layers)
    output = bound_zonotope(model, enclosure, error_propagation, bn_statistics, layers)
    ce = F.cross_entropy(output.c, target)
    if eps == 0:
        return ce
    # torch.linalg.vector_norm defines the zero-vector gradient as zero.
    norm = torch.linalg.vector_norm(torch.cat((output.G.flatten(1), output.r.flatten(1)), 1), dim=1)
    return (1 - tau) * ce + tau * norm.mean() / (output.c.shape[1] * eps)
