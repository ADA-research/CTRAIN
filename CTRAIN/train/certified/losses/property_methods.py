"""Method-specific property losses using the existing attacks and LiRPA bounds."""
import torch
from auto_LiRPA import BoundedTensor, PerturbationLpNorm
from CTRAIN.attacks import pgd_attack
from CTRAIN.bound.properties import METHODS, box_endpoints
from CTRAIN.bound.sabr import get_propagation_region
from CTRAIN.bound.util import compute_ibp_bounds, get_pgd_ptb
from CTRAIN.properties import Box
from CTRAIN.train.certified.losses.properties import property_loss
from CTRAIN.train.certified.losses.convex_combinations import _combine_mtl, _combine_exp
from CTRAIN.train.certified.losses.ibp_r import relu_area

from CTRAIN.properties.training import TRAINING_METHODS, resolve_training_options


def get_property_training_loss(wrapper, spec_index, boxes):
    """Compute the configured property loss from a PropertyModelWrapper's bound graphs and options."""
    return compute_property_training_loss(
        wrapper.bounds, spec_index, boxes, verification_model=wrapper.verification_model,
        training_method=wrapper.training_method, training_options=wrapper.training_options,
        margin=wrapper.margin, reference=wrapper.reference)


def compute_property_training_loss(bounds, spec_index, boxes, *, verification_model,
                                   training_method, training_options, margin, reference):
    """Compute the selected training objective for a batch of property regions.

    Each strategy reuses the shared satisfaction margins with its bound-based,
    adversarial, hybrid or regularised loss. The result is a scalar penalty for
    training; whole-region certification separately computes fresh sound bounds.
    """
    method, options = training_method, training_options
    def penalty(margins):
        return property_loss(margins, margin, loss=options['loss'])
    if method == 'CROWN-IBP':
        beta = options['beta']
        lower = bounds(spec_index, boxes, 'CROWN-IBP') if beta else None
        if beta < 1:
            ibp = bounds(spec_index, boxes, 'IBP')
            lower = ibp if beta == 0 else beta * lower + (1 - beta) * ibp
        return penalty(lower)
    if method in METHODS or method == 'GLS':
        bound_method = options['bound_method'] if method == 'GLS' else method
        return penalty(bounds(spec_index, boxes, bound_method,
                                      relu_upper_retention=options.get('relu_upper_retention', 1.)))
    alpha = options.get('alpha')
    if alpha == 1:
        return penalty(bounds(spec_index, boxes, 'IBP',
                                      relu_upper_retention=options['relu_upper_retention']))
    def point_margins(output, *, boolean=False):
        return bounds.point_margins(spec_index, output, boolean=boolean)
    lower, upper = box_endpoints(boxes, reference)
    nominal, half_width = (lower + upper) / 2, (upper - lower) / 2
    search = get_pgd_ptb(PerturbationLpNorm(norm=float('inf'), x_L=lower, x_U=upper),
                         nominal, options['pgd_region_factor'], eps=half_width)
    search_lower, search_upper = search.x_L, search.x_U
    attack = dict(restarts=options['pgd_restarts'], step_size=options['pgd_step_size'],
                  early_stopping=options['pgd_early_stopping'], device=nominal.device,
                  decay_factor=options['pgd_decay_factor'], decay_checkpoints=options['pgd_decay_checkpoints'],
                  loss_fn=lambda output, _: (property_loss(
                      point_margins(output),
                      loss='cross_entropy', reduction='none') if options['loss'] == 'cross_entropy'
                      else -point_margins(output).amin(1)),
                  is_success=lambda output, _: ~point_margins(output, boolean=True).all(1))
    if method == 'SABR':
        center, radius, _ = get_propagation_region(
            verification_model, nominal, None, options['subselection_ratio'],
            options['pgd_step_size'], options['pgd_steps'], options['pgd_restarts'],
            x_L=lower, x_U=upper, pgd_x_L=search_lower, pgd_x_U=search_upper,
            early_stopping=options['pgd_early_stopping'], decay_factor=options['pgd_decay_factor'],
            decay_checkpoints=options['pgd_decay_checkpoints'], loss_fn=attack['loss_fn'], is_success=attack['is_success'])
        sub_lower, sub_upper = torch.maximum(lower, center - radius), torch.minimum(upper, center + radius)
        subregions = [Box(lo, hi) for lo, hi in zip(sub_lower.detach(), sub_upper.detach())]
        return penalty(bounds(spec_index, subregions, 'IBP',
                                      relu_upper_retention=options['relu_upper_retention']))
    adversarial = pgd_attack(verification_model, nominal, None, search_lower, search_upper,
                             n_steps=options['pgd_steps'], **attack)
    output = verification_model(adversarial)
    adversarial_margins = point_margins(output)
    adversarial_loss = penalty(adversarial_margins)
    if method == 'PGD' or alpha == 0:
        return adversarial_loss
    if method == 'IBP-R':
        area = nominal.new_zeros(len(nominal))
        if options['rho']:
            raw = bounds.raw
            raw(nominal)
            ptb = PerturbationLpNorm(norm=float('inf'), x_L=lower, x_U=upper)
            compute_ibp_bounds(raw, options['relu_upper_retention'],
                x=(BoundedTensor(nominal, ptb),), method='IBP', bound_upper=True)
            area = relu_area(raw, nominal)
            if options['regularisation_mask']:
                area = area * point_margins(output.detach(), boolean=True).all(1)
        return adversarial_loss + options['rho'] * area.mean()
    lower_margins = bounds(spec_index, boxes, 'IBP',
                                  relu_upper_retention=options['relu_upper_retention'])
    robust_loss = penalty(lower_margins)
    if method == 'MTL-IBP':
        return _combine_mtl(alpha, robust_loss, adversarial_loss)
    if method == 'CC-IBP':
        return penalty(alpha * lower_margins + (1 - alpha) * adversarial_margins)
    return _combine_exp(alpha, robust_loss, adversarial_loss)
