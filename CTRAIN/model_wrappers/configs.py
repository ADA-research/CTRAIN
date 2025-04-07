from ConfigSpace import Categorical, ConfigurationSpace, EqualsCondition, Float, Integer, Constant
from ConfigSpace.types import NotSet
import math

import torch


def get_config_space(model_wrapper, epochs, eps, defaults=dict()):
    if model_wrapper.cert_train_method == 'shi':
        return build_shi_config_space(epochs, eps, defaults=defaults)
    elif model_wrapper.cert_train_method == 'crown_ibp':
        return build_crown_ibp_config_space(epochs, eps, defaults=defaults)
    elif model_wrapper.cert_train_method == 'sabr':
        return build_sabr_config_space(epochs, eps, defaults=defaults)
    elif model_wrapper.cert_train_method == 'taps':
        return build_taps_config_space(epochs, eps, defaults=defaults)
    elif model_wrapper.cert_train_method == 'staps':
        return build_staps_config_space(epochs, eps, defaults=defaults)
    elif model_wrapper.cert_train_method == 'mtl_ibp':
        return build_mtl_ibp_config_space(epochs, eps, defaults=defaults)
    else:
        assert False, 'Unknown model wrapper!'

def get_base_config(epochs, eps, defaults=dict()):
    # TODO: maybe make it possible to pass lr decay epochs as tuple as done in wrapper
    # TODO: maybe convert default torch.optim instance to string

    return ConfigurationSpace(
        name='base',
        space={
            'warm_up_epochs': Integer('warm_up_epochs', (0, 5), default=defaults.get('warm_up_epochs')),
            'ramp_up_epochs': Integer('ramp_up_epochs', (10, math.ceil(0.75*epochs)), default=defaults.get('ramp_up_epochs')),
            'lr_decay_factor': Float('lr_decay_factor', (1e-5, 0.9), log=True, default=defaults.get('lr_decay_factor')),
            'lr_decay_epoch_1': Integer('lr_decay_epoch_1', (0, math.ceil(.5 * epochs)), default=defaults.get('lr_decay_epoch_1')),
            'lr_decay_epoch_2': Integer('lr_decay_epoch_2', (0, math.ceil(.25 * epochs)), default=defaults.get('lr_decay_epoch_2')),
            'l1_reg_weight': Float('l1_reg_weight', (1e-8, 1e-4), log=True, default=defaults.get('l1_reg_weight')),
            'shi_reg_weight': Float("shi_reg_weight", (0.0, 1.0), default=defaults.get('shi_reg_weight')),
            'shi_reg_decay': Constant('shi_reg_decay', True),
            'train_eps_factor': Float("train_eps_factor", (1, 2), default=defaults.get('train_eps_factor')),
            'optimizer_func': Categorical('optimizer_func', ['adam', 'adamw', 'radam'], default=defaults.get('optimizer_func', NotSet)),
            'learning_rate': Float('learning_rate', (1e-5, 0.1), log=True, default=defaults.get('lr'))
        }
    )

def build_shi_config_space(epochs, eps, include_base_config=True, defaults=dict()):
    if include_base_config:
        config_space = get_base_config(epochs, eps, defaults=defaults)
    else:
        config_space = ConfigurationSpace()
    
    shi_config_space = ConfigurationSpace(
        name='shi',
        space={
            'start_kappa': Float('start_kappa', (0.5, 1), default=defaults.get('start_kappa')),
            'end_kappa': Float('end_kappa', (0, 1), default=defaults.get('end_kappa')) # we multiply start kappa by this factor to obtain the real end_kappa
        }
    )
    config_space.add_configuration_space('shi', shi_config_space)

    return config_space 


def build_crown_ibp_config_space(epochs, eps, include_base_config=True, defaults=dict()):
    if include_base_config:
        config_space = get_base_config(epochs, eps, defaults=defaults)
    else:
        config_space = ConfigurationSpace()

    crown_ibp_config_space = ConfigurationSpace(
        name='crown_ibp',
        space={
            'start_kappa': Float('start_kappa', (0.5, 1), default=defaults.get('start_kappa')),
            'end_kappa': Float('end_kappa', (0, 1), default=defaults.get('end_kappa')), # we multiply start kappa by this factor to obtain the real end_kappa
            'start_beta': Constant('start_beta', 1.0),
            'end_beta': Float('end_beta', (0, 1), default=defaults.get('end_beta'))
        }
    )
    config_space.add_configuration_space('crown_ibp', crown_ibp_config_space)

    return config_space 


def build_sabr_config_space(epochs, eps, include_base_config=True, defaults=dict()):
    if include_base_config:
        config_space = get_base_config(epochs, eps, defaults=defaults)
    else:
        config_space = ConfigurationSpace()

    sabr_config_space = ConfigurationSpace(
        name='sabr',
        space={
            'subselection_ratio': Float('subselection_ratio', (0.001, 0.5), log=True, default=defaults.get('subselection_ratio')),
            'pgd_steps': Integer('pgd_steps', (1, 10), default=defaults.get('pgd_steps')),
            'pgd_alpha': Float('pgd_alpha', (0.1, 2), default=defaults.get('pgd_alpha')),
            'pgd_restarts': Constant('pgd_restarts', 1), # fixed number since it can get very expensive with more restarts
            'pgd_eps_factor': Float('pgd_eps_factor', (1, 3), default=defaults.get('pgd_eps_factor')),
            # we do not optimise the pgd decay
        },

    )
    config_space.add_configuration_space('sabr', sabr_config_space)

    return config_space 

def build_taps_config_space(epochs, eps, include_base_config=True, defaults=dict()):
    if include_base_config:
        config_space = get_base_config(epochs, eps, defaults=defaults)
    else:
        config_space = ConfigurationSpace()
    
    taps_config_space = ConfigurationSpace(
        name='taps',
        space={
            'pgd_steps': Integer('pgd_steps', (1, 10), default=defaults.get('pgd_steps')),
            'pgd_alpha': Float('pgd_alpha', (0.1, 2), default=defaults.get('pgd_alpha')),
            'pgd_restarts': Constant('pgd_restarts', 1), # fixed number since it can get very expensive with more restarts
            'gradient_expansion_alpha': Float('gradient_expansion_alpha', (1, 10), default=defaults.get('gradient_expansion_alpha')),
            'block_split_point': Float('block_split_point', (0.01, 0.99), default=defaults.get('block_split_point'))
            # we do not optimise the pgd decay
        },

    )
    config_space.add_configuration_space('taps', taps_config_space)

    return config_space 

def build_staps_config_space(epochs, eps, include_base_config=True, defaults=dict()):
    config_space = build_taps_config_space(epochs, eps, defaults=defaults, include_base_config=include_base_config)

    sabr_config_space = ConfigurationSpace(
        name='sabr',
        space={
            'subselection_ratio': Float('subselection_ratio', (0.001, 0.5), log=True, default=defaults.get('subselection_ratio')),
            'pgd_steps': Integer('pgd_steps', (1, 10), default=defaults.get('pgd_steps')),
            'pgd_alpha': Float('pgd_alpha', (0.1, 2), default=defaults.get('pgd_alpha')),
            'pgd_restarts': Constant('pgd_restarts', 1), # fixed number since it can get very expensive with more restarts
            'pgd_eps_factor': Float('pgd_eps_factor', (1, 3), default=defaults.get('pgd_eps_factor')),
            # we do not optimise the pgd decay
        },

    )
    config_space.add_configuration_space('sabr', sabr_config_space)

    return config_space 


def build_mtl_ibp_config_space(epochs, eps, include_base_config=True, defaults=dict()):
    if include_base_config:
        config_space = get_base_config(epochs, eps, defaults=defaults)
    else:
        config_space = ConfigurationSpace()

    mtl_ibp_config_space = ConfigurationSpace(
        name='mtl_ibp',
        space={
            'mtl_ibp_alpha': Float('mtl_ibp_alpha', (0.001, 0.5), log=True, default=defaults.get('mtl_ibp_alpha')),
            'pgd_steps': Integer('pgd_steps', (1, 10), default=defaults.get('pgd_steps')),
            'pgd_alpha': Float('pgd_alpha', (0.1, 2), default=defaults.get('pgd_alpha')),
            'pgd_restarts': Constant('pgd_restarts', 1), # fixed number since it can get very expensive with more restarts
            'mtl_ibp_eps_factor': Float("mtl_ibp_eps_factor", (1, 3), default=defaults.get('mtl_ibp_eps_factor')),
        },

    )
    config_space.add_configuration_space('mtl_ibp', mtl_ibp_config_space)

    return config_space 

def get_combined_config_space(epoch, eps, defaults=dict(), included_methods=['shi', 'crown_ibp', 'sabr', 'mtl_ibp', 'taps', 'staps']):
    base_config_space = get_base_config(epoch, eps, defaults=defaults)
    base_config_space.add_hyperparameter(
        Categorical('cert_train_method', included_methods)
    )
    if 'shi' in included_methods:
        shi_config = build_shi_config_space(epoch, eps, include_base_config=False, defaults=defaults)
        base_config_space.add_configuration_space('shi', shi_config)
        base_config_space.add_conditions(
            [EqualsCondition(base_config_space[param], base_config_space['cert_train_method'], 'shi') for param in base_config_space if 'shi:' in param],
        )
    if 'crown_ibp' in included_methods:
        crown_ibp_config = build_crown_ibp_config_space(epoch, eps, include_base_config=False, defaults=defaults)
        base_config_space.add_configuration_space('crown_ibp', crown_ibp_config)
        base_config_space.add_conditions(
            [EqualsCondition(base_config_space[param], base_config_space['cert_train_method'], 'crown_ibp') for param in base_config_space if 'crown_ibp:' in param],
        )
    if 'sabr' in included_methods:
        sabr_config = build_sabr_config_space(epoch, eps, include_base_config=False, defaults=defaults)
        base_config_space.add_configuration_space('sabr', sabr_config)
        base_config_space.add_conditions(
            [EqualsCondition(base_config_space[param], base_config_space['cert_train_method'], 'sabr') for param in base_config_space if 'sabr:' in param],
        )
    if 'taps' in included_methods:
        taps_config = build_taps_config_space(epoch, eps, include_base_config=False, defaults=defaults)
        base_config_space.add_configuration_space('taps', taps_config)
        base_config_space.add_conditions(
            [EqualsCondition(base_config_space[param], base_config_space['cert_train_method'], 'taps') for param in base_config_space if 'taps:' in param],
        )
    if 'staps' in included_methods:
        staps_config = build_staps_config_space(epoch, eps, include_base_config=False, defaults=defaults)
        base_config_space.add_configuration_space('staps', staps_config)
        base_config_space.add_conditions(
            [EqualsCondition(base_config_space[param], base_config_space['cert_train_method'], 'staps') for param in base_config_space if 'staps:' in param],
        )
    if 'mtl_ibp' in included_methods:
        mtl_ibp_config = build_mtl_ibp_config_space(epoch, eps, include_base_config=False, defaults=defaults)
        base_config_space.add_configuration_space('mtl_ibp', mtl_ibp_config)
        base_config_space.add_conditions(
            [EqualsCondition(base_config_space[param], base_config_space['cert_train_method'], 'mtl_ibp') for param in base_config_space if 'mtl_ibp:' in param],
        )

    return base_config_space
    
    
    

