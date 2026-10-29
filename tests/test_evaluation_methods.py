"""CPU checks for alpha-CROWN soundness/state and AutoAttack pixel geometry."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from auto_LiRPA import BoundedModule, PerturbationLpNorm
from CTRAIN.bound import bound_alpha_crown
from CTRAIN.eval import eval_alpha_crown, eval_certified, eval_autoattack, eval_adversarial
from CTRAIN.model_wrappers.model_wrapper import CTRAINWrapper


def check():
    torch.manual_seed(3)
    native = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 2))
    data = torch.tensor([[.4]])
    model = BoundedModule(native, data, device='cpu')
    model.train()
    parameters = list(model.parameters())
    for p in parameters:
        p.grad = torch.ones_like(p)
    gradients = [p.grad.clone() for p in parameters]
    options = copy.deepcopy(model.bound_opts)
    modes = [m.training for m in model.modules()]
    region = PerturbationLpNorm(x_L=data-.3, x_U=data+.3)
    with torch.no_grad():
        lower, upper = bound_alpha_crown(model, region, data, torch.tensor([0]),
                                         n_classes=2, bound_upper=True, alpha_iterations=3)
    grid = torch.linspace(.1, .7, 101).reshape(-1, 1)
    outputs = native(grid)
    margins = outputs[:, 0] - outputs[:, 1]
    assert lower.item() <= margins.min().item() + 1e-5
    assert upper.item() >= margins.max().item() - 1e-5
    assert model.bound_opts == options and [m.training for m in model.modules()] == modes
    assert all(p.requires_grad and torch.equal(p.grad, g) for p, g in zip(parameters, gradients))
    with patch.object(model, 'compute_bounds', side_effect=RuntimeError('test failure')):
        try:
            bound_alpha_crown(model, region, data, torch.tensor([0]), alpha_iterations=3)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Expected bound failure')
    assert model.bound_opts == options and model.training
    assert all(p.requires_grad and torch.equal(p.grad, g) for p, g in zip(parameters, gradients))
    loader = DataLoader(TensorDataset(data.repeat(2, 1), torch.zeros(2, dtype=torch.long)), batch_size=2)
    loader.min, loader.max = torch.tensor(0.), torch.tensor(1.)
    loader.std, loader.normalised = torch.tensor(1.), False
    _, total, records = eval_alpha_crown(model, .1, loader, 2, 1, 'cpu', 3)
    assert total == 1 and list(records) == [0] and model.training
    accuracy, records = eval_certified(model, loader, .1, 2, 1,
                                       ['IBP', 'ALPHA-CROWN'], 'cpu', alpha_iterations=3)
    assert 0 <= accuracy <= 1 and model.training

    class PixelClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.))

        def forward(self, x):
            z = x.flatten(1).mean(1) * self.scale
            return torch.stack([.5-z, z-.5], dim=1)

    classifier = PixelClassifier().train()
    classifier.scale.grad = torch.tensor(7.)
    # Channelwise normalization must not change the raw-pixel threat set.
    mean = torch.tensor([.1, .3]).reshape(2, 1, 1)
    std = torch.tensor([.2, .4]).reshape(2, 1, 1)
    class Normalized(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = classifier

        def forward(self, x):
            return classifier(x*std + mean)

    normalized = Normalized().train()
    pixels = torch.tensor([.55, .8, .55]).reshape(3, 1, 1, 1).expand(3, 2, 1, 1)
    loader = DataLoader(TensorDataset((pixels-mean)/std, torch.ones(3, dtype=torch.long)), batch_size=2)
    loader.min, loader.max = -mean/std, (1-mean)/std
    class Attack:
        def __init__(self, model, **kwargs):
            assert abs(kwargs['eps']-.1) < 1e-6 and kwargs['norm'] == 'Linf'
            self.model = model

        def run_standard_evaluation(self, x, y, bs):
            assert x.min() >= 0 and x.max() <= 1
            assert torch.allclose(self.model(x), classifier(x))
            return (x-.1).clamp(0, 1)

    with patch.dict(sys.modules, autoattack=SimpleNamespace(AutoAttack=Attack)):
        accuracy, mask, records = eval_adversarial(
            normalized, loader, .1/std, True, device='cpu', test_samples=2, method='AUTOATTACK')
        assert accuracy == .5 and mask.tolist() == [True, False] and list(records) == [0, 1]
        assert normalized.training and classifier.training and classifier.scale.requires_grad
        assert classifier.scale.grad.item() == 7
        with patch.object(Attack, 'run_standard_evaluation', side_effect=RuntimeError('attack failed')):
            try:
                eval_autoattack(normalized, loader, .1/std, device='cpu')
            except RuntimeError:
                pass
            else:
                raise AssertionError('Expected attack failure')
        assert normalized.training and classifier.scale.requires_grad and classifier.scale.grad.item() == 7
        try:
            eval_autoattack(normalized, loader, .1, device='cpu')
        except ValueError:
            pass
        else:
            raise AssertionError('Unequal raw radii must be rejected')
    # Wrapper forwarding retains the public three-accuracy interface.
    wrapper = CTRAINWrapper(nn.Sequential(nn.Flatten(), nn.Linear(2, 2)), .1,
                            (2, 1, 1), device='cpu')
    loader.std, loader.normalised = std.flatten(), True
    with patch('CTRAIN.model_wrappers.model_wrapper.eval_model', return_value=(.8, .5, .6)) as evaluate:
        assert wrapper.evaluate(loader, eval_method='ALPHA-CROWN', adversarial_method='AUTOATTACK',
                                alpha_iterations=3) == (.8, .5, .6)
        assert evaluate.call_args.kwargs['adversarial_method'] == 'AUTOATTACK'
        assert evaluate.call_args.kwargs['alpha_iterations'] == 3
    print('Evaluation method checks passed')


if __name__ == '__main__':
    check()
