"""CPU regression: paper objectives/gradients, attack state, wrappers and HPO."""
from unittest.mock import patch
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from auto_LiRPA import BoundedModule, PerturbationLpNorm
from CTRAIN import MTLIBPModelWrapper, CCIBPModelWrapper, ExpIBPModelWrapper, ExpressiveIBPModelWrapper
from CTRAIN.attacks import pgd_attack
from CTRAIN.util import construct_c
from CTRAIN.train.certified.losses import get_mtl_ibp_loss, get_cc_ibp_loss, get_exp_ibp_loss
from CTRAIN.train.certified.util import synchronise_bn
from CTRAIN.model_wrappers.configs import get_config_space, get_combined_config_space
from CTRAIN.train.certified.losses import get_ibp_loss
from CTRAIN.bound import bound_ibp

PATH = 'CTRAIN.train.certified.losses.convex_combinations'


def check():
    assert all(cls.__bases__ == (ExpressiveIBPModelWrapper,)
               for cls in (MTLIBPModelWrapper, CCIBPModelWrapper, ExpIBPModelWrapper))
    assert not issubclass(CCIBPModelWrapper, MTLIBPModelWrapper)
    assert not issubclass(ExpIBPModelWrapper, MTLIBPModelWrapper)
    try:
        ExpressiveIBPModelWrapper(None, (1, 2, 1, 1), .1, 10, device='cpu')
    except TypeError:
        pass
    else:
        raise AssertionError('common wrapper must require a specialized loss')
    from CTRAIN.train.certified import mtl_ibp_train_model, cc_ibp_train_model, exp_ibp_train_model
    for entry, method, loss in ((mtl_ibp_train_model, 'mtl_ibp', get_mtl_ibp_loss),
                               (cc_ibp_train_model, 'cc_ibp', get_cc_ibp_loss),
                               (exp_ibp_train_model, 'exp_ibp', get_exp_ibp_loss)):
        with patch(f'CTRAIN.train.certified.{method}.expressive_ibp_train_model',
                   return_value='delegated') as shared:
            assert entry(None, None, None, alpha=.37) == 'delegated'
            assert shared.call_args.kwargs['method'] == method
            assert shared.call_args.kwargs['loss_func'] is loss
            assert shared.call_args.kwargs['alpha'] == .37
    torch.set_num_threads(1)
    torch.manual_seed(12)
    x = torch.tensor([[.1, .6], [.8, .3], [.4, .9]])
    y = torch.tensor([2, 0, 1])
    adv = x + torch.tensor([[.03, -.04], [-.02, .05], [.04, -.03]])
    original = nn.Sequential(nn.Linear(2, 3)).eval()
    model = BoundedModule(original, x, device='cpu', bound_opts={'param': {'auto_requires_grad': False}})
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    criterion = nn.CrossEntropyLoss(reduction='none')
    c = construct_c(x, y, 3)
    weight, bias = original[0].weight, original[0].bias
    margin_weight = c @ weight
    exact_lb = (margin_weight * x.unsqueeze(1)).sum(-1) + (c @ bias) - .1 * margin_weight.abs().sum(-1)
    adv_margin = (c @ original(adv).unsqueeze(-1)).squeeze(-1)
    ce = lambda margins: torch.logsumexp(torch.cat([torch.zeros(3, 1), -margins], dim=1), dim=1).mean()
    robust, adversarial = ce(exact_lb), criterion(original(adv), y).mean()
    for func in (get_mtl_ibp_loss, get_cc_ibp_loss, get_exp_ibp_loss):
        for alpha in (0., .37, 1.):
            region = PerturbationLpNorm(eps=.1, x_L=x-.1, x_U=x+.1)
            with patch(PATH + '.pgd_attack', return_value=adv):
                loss, lb, ub, _, _ = func(model, original, region, x, y, 3, criterion, alpha,
                                          return_bounds=True, return_stats=True, device='cpu')
            expected = (ce(alpha*exact_lb+(1-alpha)*adv_margin) if func is get_cc_ibp_loss
                        else robust**alpha * adversarial**(1-alpha) if func is get_exp_ibp_loss
                        else alpha * robust + (1-alpha) * adversarial)
            torch.testing.assert_close(lb, exact_lb)
            torch.testing.assert_close(loss, expected)
            actual_grad = torch.autograd.grad(loss, (weight, bias), retain_graph=True)
            expected_grad = torch.autograd.grad(expected, (weight, bias), retain_graph=True)
            for actual, reference in zip(actual_grad, expected_grad):
                torch.testing.assert_close(actual, reference)
            assert not model.training and not original.training
            assert all(p.grad is None for p in original.parameters())
        for alpha in (-.1, 1.1, float('nan'), float('inf')):
            with patch(PATH + '.pgd_attack') as attack:
                try:
                    func(model, original, region, x, y, 3, criterion, alpha, device='cpu')
                except ValueError:
                    pass
                else:
                    raise AssertionError('invalid coefficient accepted')
                attack.assert_not_called()
    # The attack runs in eval, restores mixed caller modes and frozen parameters
    # on failure, and does not touch accumulated parameter gradients.
    model.train()
    original.train()
    original[0].eval()
    weight.requires_grad_(False)
    bias.grad = torch.ones_like(bias)
    modes = [(m, m.training) for root in (model, original) for m in root.modules()]
    flags = [(p, p.requires_grad) for root in (model, original) for p in root.parameters()]
    def fail(**kwargs):
        assert not model.training and not original.training
        weight.requires_grad_(True)
        raise RuntimeError('attack failed')
    with patch(PATH + '.pgd_attack', side_effect=fail):
        try:
            get_cc_ibp_loss(model, original, region, x, y, 3, criterion, .5, device='cpu')
        except RuntimeError as error:
            assert str(error) == 'attack failed'
    assert all(m.training == state for m, state in modes)
    assert all(p.requires_grad == state for p, state in flags)
    assert torch.equal(bias.grad, torch.ones_like(bias))
    # Confident CE rounds to zero: the geometric objective must retain finite gradients.
    weight.requires_grad_(True)
    zero = bias.sum() * 0
    with patch(PATH + '.pgd_attack', return_value=adv), \
         patch(PATH + '.get_ibp_loss', return_value=(zero, exact_lb, exact_lb)), \
         patch.object(criterion, 'forward', return_value=zero):
        loss, = get_exp_ibp_loss(model, original, region, x, y, 3, criterion, .5, device='cpu')
        assert torch.isfinite(loss)
        assert torch.isfinite(torch.autograd.grad(loss, bias)[0]).all()

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(), nn.Linear(2, 3))
        def forward(self, data):
            return self.layers(data)
    images = x.reshape(3, 2, 1, 1)
    loader = DataLoader(TensorDataset(images, y), batch_size=3)
    loader.normalised = False
    loader.mean = torch.zeros(2)
    loader.std = torch.tensor([2., 3.])  # Must not divide radius for unnormalised data.
    loader.min, loader.max = torch.tensor(0.), torch.tensor(1.)
    for wrapper_type, method in ((MTLIBPModelWrapper, 'mtl_ibp'), (CCIBPModelWrapper, 'cc_ibp'), (ExpIBPModelWrapper, 'exp_ibp')):
        wrapper = wrapper_type(Toy(), (1, 2, 1, 1), .1, 10, warm_up_epochs=0,
                               ramp_up_epochs=8, device='cpu', population_bn=True,
                               pgd_steps=1, pgd_eps_factor=2., relu_upper_retention=.8,
                               **{method + '_alpha': .37})
        assert wrapper.cert_train_method == method
        before = [p.detach().clone() for p in wrapper.parameters()]
        with patch('CTRAIN.train.certified.expressive_ibp.synchronise_bn', wraps=synchronise_bn) as sync, \
             patch(PATH + '.pgd_attack', wraps=pgd_attack) as attack, \
             patch(PATH + '.get_ibp_loss', wraps=get_ibp_loss) as bounds:
            with patch('CTRAIN.train.certified.mtl_ibp.mtl_ibp_train_model',
                       side_effect=AssertionError('wrapper must use the neutral trainer')):
                wrapper.train_model(loader, start_epoch=9, end_epoch=10)
            assert sync.call_count == 1
            args = attack.call_args.kwargs
            torch.testing.assert_close(args['x_L'], (images-.2).clamp(0, 1))
            torch.testing.assert_close(args['x_U'], (images+.2).clamp(0, 1))
            nominal = bounds.call_args.kwargs
            assert nominal['relu_upper_retention'] == .8
            torch.testing.assert_close(nominal['ptb'].x_L, (images-.1).clamp(0, 1))
            torch.testing.assert_close(nominal['ptb'].x_U, (images+.1).clamp(0, 1))
        # Ordinary certification must agree with explicit sound propagation after training.
        wrapper.eval()
        region = PerturbationLpNorm(x_L=(images-.1).clamp(0, 1), x_U=(images+.1).clamp(0, 1))
        lb, _ = bound_ibp(wrapper.bounded_model, region, images, y, 3)
        reference_lb, _ = bound_ibp(wrapper.bounded_model, region, images, y, 3, relu_upper_retention=1.)
        torch.testing.assert_close(lb, reference_lb)
        assert any(not torch.equal(a, b) for a, b in zip(before, wrapper.parameters()))
        assert all(torch.isfinite(p).all() for p in wrapper.parameters())
        for name, value in wrapper.bounded_model.named_buffers():
            assert torch.equal(value, dict(wrapper.original_model.named_buffers())[name])
        config = get_config_space(wrapper, 10, .1, defaults={'pgd_eps_factor': 2.}).get_default_configuration()
        assert config[method + ':pgd_eps_factor'] == 2.
        assert method + ':' + method + '_eps_factor' not in config
        assert method + ':' + method + '_alpha' in config
        # Exercise inherited HPO recreation with actual bounded models,
        # and method-specific coefficient/options; expensive evaluation is mocked.
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'nets').mkdir()
            with patch.object(wrapper_type, '_evaluate_hpo_model', return_value=(.5, .4, .3)), \
                 patch.object(wrapper_type, 'train_model', autospec=True) as train:
                wrapper._hpo_runner(config, 0, 10, loader, loader, directory)
                recreated = train.call_args.args[0]
                assert recreated.pgd_eps_factor == config[method + ':pgd_eps_factor']
                assert type(recreated) is wrapper_type
                assert recreated.population_bn and recreated.relu_upper_retention == .8
                assert getattr(recreated, method + '_alpha') == config[method + ':' + method + '_alpha']
        try:
            wrapper_type(None, (1, 2, 1, 1), .1, 10, device='cpu', **{method+'_alpha': -1})
        except ValueError:
            pass
        else:
            raise AssertionError('validation must precede tracing')
    combined = get_combined_config_space(10, .1, included_methods=['cc_ibp', 'exp_ibp'])
    for config in combined.sample_configuration(10):
        method = config['cert_train_method']
        assert method + ':' + method + '_alpha' in config, dict(config)
        assert method + ':pgd_eps_factor' in config
        other = 'exp_ibp' if method == 'cc_ibp' else 'cc_ibp'
        assert all(not key.startswith(other + ':') for key in config)
    print('Expressive loss formula, gradient, state, training and HPO checks passed')


if __name__ == '__main__':
    check()
