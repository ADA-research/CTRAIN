"""General-property training strategies and their validated scalar parameters."""
import copy
import math


BOUND_METHODS = ('IBP', 'CROWN-IBP', 'CROWN')
TRAINING_METHODS = (*BOUND_METHODS, 'MTL-IBP', 'CC-IBP', 'EXP-IBP', 'SABR', 'PGD', 'IBP-R', 'GLS')
_UNSUPPORTED = {
    'TAPS': 'latent general-property search and splits are not defined',
    'STAPS': 'latent general-property search and splits are not defined',
    'SET-BASED': 'the enclosure-size objective requires a separate property formulation',
    'CC-DIST': 'teacher-feature distillation requires a separate property formulation',
}
_ATTACK = dict(pgd_steps=8, pgd_step_size=.5, pgd_restarts=1, pgd_early_stopping=False,
               pgd_region_factor=1., pgd_decay_factor=.1, pgd_decay_checkpoints=[])


def resolve_training_options(method, options=None):
    """Resolve and validate the options for a general-property training strategy.

    The chosen strategy reuses satisfaction margins with its own bound-based,
    adversarial, hybrid or smoothed objective. Unknown options and unsupported
    property-method adaptations are rejected before bound-graph conversion.

    Args:
        method (str): Training-method name, normalised to uppercase with hyphens.
        options (dict, optional): Method-specific overrides; copied before defaults are
            added.

    Returns:
        (tuple): Canonical method name and a validated independent options dictionary.
    """
    if not isinstance(method, str):
        raise ValueError('training_method must be a method name')
    method = method.upper().replace('_', '-')
    if method in _UNSUPPORTED:
        raise ValueError(f'General-property {method} training is not supported: {_UNSUPPORTED[method]}')
    if method not in TRAINING_METHODS:
        raise ValueError(f'training_method must be one of {TRAINING_METHODS}')
    defaults = dict(loss='hinge', gradient_clip=None)
    if method == 'CROWN-IBP':
        defaults['beta'] = 1.
    if method in ('IBP', 'MTL-IBP', 'CC-IBP', 'EXP-IBP', 'SABR', 'IBP-R'):
        defaults['relu_upper_retention'] = 1.
    if method in ('MTL-IBP', 'CC-IBP', 'EXP-IBP', 'SABR', 'PGD', 'IBP-R'):
        defaults.update(copy.deepcopy(_ATTACK))
    if method in ('MTL-IBP', 'CC-IBP', 'EXP-IBP'):
        defaults['alpha'] = .5
    if method == 'SABR':
        defaults['subselection_ratio'] = .1
    if method == 'IBP-R':
        defaults.update(rho=2e-4, regularisation_mask=False)
    if method == 'GLS':
        defaults.update(estimator='rgs', samples=2, sigma=1e-3, sigma_lr=1e-3,
                        bound_method='CROWN', sigma_decay_milestones=[], sigma_decay_factor=.4)
    if options is not None and not isinstance(options, dict):
        raise ValueError('training_options must be a dictionary')
    options = copy.deepcopy(options or {})
    if any(not isinstance(key, str) for key in options):
        raise ValueError('Training option keys must be strings')
    unknown = set(options) - defaults.keys()
    if unknown:
        raise ValueError(f'{method} does not accept training options {sorted(unknown)}; allowed: {sorted(defaults)}')
    options = dict(defaults, **options)
    if options['loss'] not in ('hinge', 'cross_entropy'):
        raise ValueError('Property loss must be hinge or cross_entropy')
    for key in ('alpha', 'beta', 'subselection_ratio', 'rho', 'sigma', 'sigma_lr', 'sigma_decay_factor',
                'pgd_step_size', 'pgd_decay_factor', 'pgd_region_factor', 'relu_upper_retention', 'gradient_clip'):
        if key in options:
            v = options[key]
            if v is None and key == 'gradient_clip':
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f'{key} must be finite numeric')
    for key in ('alpha', 'beta', 'subselection_ratio'):
        if key in options and not 0 <= options[key] <= 1:
            raise ValueError(f'{key} must be in [0, 1]')
    for key in ('pgd_steps', 'pgd_restarts', 'samples'):
        if key in options and (type(options[key]) is not int or options[key] < (0 if key == 'pgd_steps' else 1)):
            raise ValueError(f'Invalid {key}')
    for key in ('pgd_early_stopping', 'regularisation_mask'):
        if key in options and type(options[key]) is not bool:
            raise ValueError(f'{key} must be bool')
    for key in ('pgd_step_size', 'rho', 'sigma_lr'):
        if key in options and options[key] < 0:
            raise ValueError(f'{key} must be nonnegative')
    for key in ('pgd_decay_factor', 'sigma', 'gradient_clip'):
        if key in options and options[key] is not None and options[key] <= 0:
            raise ValueError(f'{key} must be positive')
    retention = options.get('relu_upper_retention', 1.)
    if not 0 < retention <= 1 or options.get('pgd_region_factor', 1.) <= 0:
        raise ValueError('ReLU retention must be in (0,1] and PGD region factor positive')
    for key in ('pgd_decay_checkpoints', 'sigma_decay_milestones'):
        if key in options:
            if not isinstance(options[key], (list, tuple)) or any(type(n) is not int or n < (1 if key.startswith('sigma') else 0) for n in options[key]):
                raise ValueError(f'Invalid {key}')
            options[key] = list(options[key])
    if method == 'GLS':
        if options['estimator'] not in ('rgs', 'pgpe') or options['estimator'] == 'pgpe' and options['samples'] % 2:
            raise ValueError('GLS requires rgs or pgpe; PGPE samples must be even')
        if options['bound_method'] not in BOUND_METHODS or not 0 < options['sigma_decay_factor'] <= 1:
            raise ValueError('Invalid GLS bound method or scale decay')
        if options['estimator'] == 'pgpe' and options['sigma_decay_milestones']:
            raise ValueError('PGPE learns its scale; milestones are for RGS')
    return method, options
