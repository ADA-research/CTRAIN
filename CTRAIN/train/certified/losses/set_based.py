"""Set-based center classification and enclosure-size objective."""
import math
import torch
from torch.nn import functional as F
from CTRAIN.bound.zonotope import Zonotope, box_zonotope, bound_zonotope
from CTRAIN.util.util import preserve_model_state


def validate_set_options(tau, input_set, num_generators, error_propagation,
                         propagation_batch_size, pgd_eps_factor):
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
                    num_generators=8, pgd_eps_factor=1.):
    """Detached randomized FGSM directions, coordinatewise fitted to the nominal box."""
    if type(num_generators) is not int or num_generators <= 0 or not math.isfinite(pgd_eps_factor) or pgd_eps_factor <= 0:
        raise ValueError('Invalid attack generator count or PGD epsilon factor')
    lower, upper = (data - radius).clamp(data_min, data_max), (data + radius).clamp(data_min, data_max)
    search_radius = radius * pgd_eps_factor
    search_lower = (data - search_radius).clamp(data_min, data_max)
    search_upper = (data + search_radius).clamp(data_min, data_max)
    positions = []
    with preserve_model_state(model), torch.enable_grad():
        model.eval()
        for _ in range(num_generators):
            point = (search_lower + torch.rand_like(data) * (search_upper - search_lower)).detach().requires_grad_(True)
            gradient, = torch.autograd.grad(F.cross_entropy(model(point), target), point)
            point = (point + search_radius * gradient.sign()).clamp(search_lower, search_upper)
            positions.append(point.detach().clamp(lower, upper))
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
    """eps is the scheduled radius BEFORE normalization; radius is in model units."""
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
                                    num_generators, pgd_eps_factor)
    output = bound_zonotope(model, enclosure, error_propagation, bn_statistics, layers)
    ce = F.cross_entropy(output.c, target)
    if eps == 0:
        return ce
    # torch.linalg.vector_norm defines the zero-vector gradient as zero.
    norm = torch.linalg.vector_norm(torch.cat((output.G.flatten(1), output.r.flatten(1)), 1), dim=1)
    return (1 - tau) * ce + tau * norm.mean() / (output.c.shape[1] * eps)
