"""Set-based training adapters around auto_LiRPA's forward operators.

The training loss needs the output center and the norm of its generators and
independent error radii. Concrete lower/upper bounds from
``BoundedModule.compute_bounds(method='Forward')`` do not retain that information.
Attack inputs also have correlated generators; replacing them with a box changes
the training set, and the installed PerturbationLinear rejects forward mode.

Keep LiRPA's LinearBound representation and layer computations, with local
adapters for centered generator coordinates, supplied clean BatchNorm statistics,
and the numerical/autograd gaps explained below. Ordinary full-box certification
uses compute_bounds(method='Forward') through CTRAIN.eval.eval_forward.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from auto_LiRPA.linear_bound import LinearBound
from auto_LiRPA.operators import (BoundLinear, BoundConv, BoundBatchNormalization,
                                 BoundRelu, BoundFlatten, BoundIdentity)


@dataclass
class Zonotope:
    """Points c + sum_q G[:, q] * xi_q + r * eta, with xi_q, eta in [-1, 1].

    c and nonnegative r have shape [batch, ...]; G has shape [batch, Q, ...].
    Each generator's coefficient is shared across coordinates, preserving their
    correlation; eta varies independently in each coordinate.
    """

    c: torch.Tensor
    G: torch.Tensor
    r: torch.Tensor

    def bounds(self):
        radius = self.G.abs().sum(1) + self.r
        return self.c - radius, self.c + radius


def sequential_layers(model):
    """Reject branches, functional operations, and unsupported layers up front."""
    # Trace the actual forward graph: listing children can silently omit computation.
    graph = torch.fx.symbolic_trace(model)
    layers, previous, seen = [], None, set()
    supported = (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d,
                 nn.ReLU, nn.Flatten, nn.Identity)
    for node in graph.graph.nodes:
        if node.op == 'placeholder' and previous is None:
            previous = node
        elif node.op == 'call_module' and node.args == (previous,) and not node.kwargs:
            layer = graph.get_submodule(node.target)
            if type(layer) not in supported or id(layer) in seen:
                raise ValueError(f'Unsupported or reused set-based layer: {node.target}')
            if isinstance(layer, nn.Conv2d) and layer.padding_mode != 'zeros':
                raise ValueError('Set propagation requires zero convolution padding')
            if isinstance(layer, nn.Flatten) and (layer.start_dim != 1 or layer.end_dim != -1):
                raise ValueError('Set propagation supports Flatten(1, -1) only')
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)) and not layer.track_running_stats:
                raise ValueError('Certification requires BatchNorm running statistics')
            layers.append(layer)
            seen.add(id(layer))
            previous = node
        elif node.op == 'output' and node.args == (previous,):
            pass
        else:
            raise ValueError(f'Set propagation requires a sequential graph: {node.format_node()}')
    if not layers:
        raise ValueError('Set propagation requires at least one supported layer')
    return layers


def diagonal_generators(radius):
    """Represent each independent interval coordinate as its own generator."""
    return torch.diag_embed(radius.flatten(1)).reshape(
        len(radius), radius[0].numel(), *radius.shape[1:])


def box_zonotope(lower, upper):
    if lower.shape != upper.shape or not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or (lower > upper).any():
        raise ValueError('Box endpoints must be finite, ordered, and have matching shapes')
    return Zonotope((lower + upper) / 2, diagonal_generators((upper - lower) / 2),
                    torch.zeros_like(lower))


def _constant(value):
    # Exact in the perturbation set, but still differentiable in model parameters.
    return LinearBound(lb=value, ub=value, lower=value, upper=value)


def _split_weight(value):
    # LiRPA's positive/negative clamps each differentiate as 1 at zero.
    # With equal lower/upper coefficients this doubles the affine derivative.
    # Preserve all values; halve only the gradient at exactly zero to correct it.
    return torch.where(value == 0, value * .5, value)


class _SetBasedRelu(BoundRelu):
    @staticmethod
    def _relu_upper_bound(lower, upper, leaky_alpha):
        slope, intercept = BoundRelu._relu_upper_bound(lower, upper, leaky_alpha)
        # LiRPA's 1e-8 denominator clamp changes the same-slope relaxation for
        # tiny unstable intervals; preserve u/(u-l) and its training gradients.
        tiny = (lower < 0) & (upper > 0) & (upper - lower < 1e-8)
        width = torch.where(tiny, upper - lower, torch.ones_like(lower))
        return (torch.where(tiny, upper / width, slope),
                torch.where(tiny, -lower * upper / width, intercept))


def _forward_operator(layer, linear, statistics):
    """Use live model parameters and LiRPA computations without a second model."""
    dim = linear.lw.shape[1]
    if isinstance(layer, nn.Linear):
        args = [linear, _constant(_split_weight(layer.weight))]
        if layer.bias is not None:
            args.append(_constant(layer.bias))
        return BoundLinear({'transB': 1}).bound_forward(dim, *args)
    if isinstance(layer, nn.Conv2d):
        padding = (0, 0) if isinstance(layer.padding, str) else layer.padding
        attr = dict(kernel_shape=layer.kernel_size, pads=(*padding, *padding),
                    strides=layer.stride, dilations=layer.dilation, group=layer.groups)
        args = [linear, _constant(layer.weight)]
        if layer.bias is not None:
            args.append(_constant(layer.bias))
        operator = BoundConv(attr, inputs=[BoundIdentity()] * len(args))
        operator.padding = layer.padding
        return operator.bound_forward(dim, *args)
    if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if statistics is None:
            if layer.training:
                raise ValueError('Training BatchNorm requires clean batch statistics')
            mean, variance = layer.running_mean, layer.running_var
        else:
            mean, variance = statistics[layer]
        weight = layer.weight if layer.affine else torch.ones_like(mean)
        bias = layer.bias if layer.affine else torch.zeros_like(mean)
        # The trainer computes differentiable clean statistics once per chunk.
        # Reuse them without buffer updates; evaluation uses running statistics.
        operator = BoundBatchNormalization(dict(epsilon=layer.eps, momentum=.9),
                                           [], 0, {}, training=False)
        split_weight = _split_weight(weight)
        # The clamp correction belongs to the signed split, not the mean offset:
        # keep d(-gamma * mean / std)/d(gamma) whole even when gamma is zero.
        bias = bias + (split_weight - weight) * mean * torch.rsqrt(variance + layer.eps)
        return operator.bound_forward(dim, linear, _constant(split_weight),
                                      _constant(bias), _constant(mean), _constant(variance))
    if isinstance(layer, nn.ReLU):
        # LiRPA needs concrete preactivation bounds to choose the relaxation.
        radius = linear.lw.abs().sum(1)
        linear.lower, linear.upper = linear.lb - radius, linear.ub + radius
        return _SetBasedRelu(options={'activation_bound_option': 'same-slope'}).bound_forward(dim, linear)
    if isinstance(layer, nn.Flatten):
        operator = BoundFlatten({'axis': 1})
        operator.input_shape = linear.lb.shape
        return operator.bound_forward(dim, linear)
    if isinstance(layer, nn.Identity):
        return BoundIdentity().bound_forward(dim, linear)
    raise ValueError(f'Unsupported set-based layer: {type(layer).__name__}')


def bound_zonotope(model, zonotope, error_propagation='interval', bn_statistics=None, layers=None):
    """Propagate the full training representation, retaining its autograd graph.

    Same-slope ReLUs keep lw=uw=G. With generator coordinates in [-1, 1],
    LiRPA's affine biases are lb=c-r and ub=c+r; they are not concrete bounds.
    Interval mode lets LiRPA carry independent errors in those biases. Generator
    mode converts errors into new correlated directions after each ReLU.
    """
    if error_propagation not in ('interval', 'generators'):
        raise ValueError('error_propagation must be interval or generators')
    layers = sequential_layers(model) if layers is None else layers
    c, G, r = zonotope.c, zonotope.G, zonotope.r
    if error_propagation == 'generators' and (r != 0).any():
        G = torch.cat((G, diagonal_generators(r)), 1)
        r = torch.zeros_like(r)
    linear = LinearBound(G, c - r, G, c + r)
    for layer in layers:
        linear = _forward_operator(layer, linear, bn_statistics)
        # Same-slope propagation keeps both coefficient tensors equal. Share them.
        linear.uw = linear.lw
        if error_propagation == 'generators' and isinstance(layer, nn.ReLU):
            # Installed dynamic-forward lacks BN support and ReLU scatter backward;
            # append errors locally so later layers retain them with working autograd.
            c, r = (linear.lb + linear.ub) / 2, (linear.ub - linear.lb) / 2
            G = linear.lw
            if (r != 0).any():
                G = torch.cat((G, diagonal_generators(r)), 1)
            linear = LinearBound(G, c, G, c)
    return Zonotope((linear.lb + linear.ub) / 2, linear.lw, (linear.ub - linear.lb) / 2)


def zonotope_margins(zonotope, labels):
    """Correlated lower bounds on true-class minus competing-class logits."""
    c, G, r = zonotope.c, zonotope.G, zonotope.r
    if c.ndim != 2:
        raise ValueError('Certification requires a batch of class logits')
    index = labels[:, None]
    gy = G.gather(2, index[:, None].expand(-1, G.shape[1], 1))
    # Shared generator contributions can cancel; independent radii must be added.
    margins = c.gather(1, index) - c - (gy - G).abs().sum(1) - r.gather(1, index) - r
    return margins.masked_fill(F.one_hot(labels, c.shape[1]).bool(), float('inf'))
