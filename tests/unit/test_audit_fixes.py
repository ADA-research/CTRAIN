"""Audit regressions: truthful metrics, evaluation state, budgets and checkpoint safety."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from auto_LiRPA import BoundedModule, PerturbationLpNorm

from CTRAIN.eval import eval_acc, eval_adversarial, eval_certified, eval_autoattack, eval_epoch
from CTRAIN.train.certified.losses import get_ibp_loss, get_crown_ibp_loss, get_sabr_loss, get_taps_loss
from CTRAIN.util.util import validate_test_samples


@pytest.mark.parametrize('budget,expected', [(1, 1), (1.0, 1), (3., 3), (float('inf'), float('inf'))])
def test_integer_sample_budgets(budget, expected):
    assert validate_test_samples(budget) == expected


@pytest.mark.parametrize('budget', [0, -1, 1.5, True, float('nan'), -float('inf'), None])
@pytest.mark.parametrize('kind', ['clean', 'certified', 'pgd', 'autoattack'])
def test_invalid_sample_budget_precedes_evaluation(budget, kind):
    model = nn.Linear(1, 2)
    with patch.object(model, 'forward', side_effect=AssertionError('evaluated invalid input')):
        with pytest.raises(ValueError, match='test_samples'):
            if kind == 'clean':
                eval_acc(model, None, budget, 'cpu')
            elif kind == 'certified':
                eval_certified(model, None, .1, test_samples=budget, device='cpu')
            elif kind == 'autoattack':
                eval_autoattack(model, None, .1, test_samples=budget, device='cpu')
            else:
                eval_adversarial(model, None, .1, test_samples=budget, device='cpu')


@pytest.mark.parametrize('bounded', [False, True])
@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('kind', ['clean', 'pgd', 'autoattack', 'epoch'])
def test_standalone_evaluation_preserves_modes_flags_buffers_gradients(bounded, fail, kind, tmp_path):
    data = torch.tensor([[.2], [.6], [.8]])
    target = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(data, target), batch_size=2)
    loader.min, loader.max = torch.tensor(0.), torch.tensor(1.)
    model = nn.Sequential(nn.Linear(1, 2), nn.BatchNorm1d(2), nn.Linear(2, 2)).eval()
    if bounded:
        model = BoundedModule(model, data[:1], device='cpu')
    model.train()
    next(m for m in model.modules() if isinstance(m, nn.Linear) or type(m).__name__ == 'BoundLinear').eval()
    for index, parameter in enumerate(model.parameters()):
        parameter.requires_grad_(index % 2 == 0)
        parameter.grad = torch.ones_like(parameter)
    modes = {m: m.training for m in model.modules()}
    flags = {p: p.requires_grad for p in model.parameters()}
    buffers = {name: value.clone() for name, value in model.named_buffers()}

    class Attack:
        def __init__(self, model, **kwargs):
            self.model = model

        def run_standard_evaluation(self, x, y, bs):
            self.model(x)
            return x

    def evaluate():
        if kind == 'clean':
            return eval_acc(model, loader, test_samples=1., device='cpu')
        if kind == 'epoch':
            return eval_epoch(model, loader, 0., 2, device='cpu', test_samples=1., results_path=tmp_path)[0]
        return eval_adversarial(model, loader, .1, test_samples=1., device='cpu',
                                restarts=1, n_steps=1, method='AUTOATTACK' if kind == 'autoattack' else 'PGD')

    with patch.dict(sys.modules, autoattack=SimpleNamespace(AutoAttack=Attack)):
        if fail:
            with patch.object(model, 'forward', side_effect=RuntimeError('injected')):
                with pytest.raises(RuntimeError, match='injected'):
                    evaluate()
        else:
            assert 0 <= evaluate() <= 1
    assert all(m.training == mode for m, mode in modes.items())
    assert all(p.requires_grad == flag for p, flag in flags.items())
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    for name, before in buffers.items():
        torch.testing.assert_close(dict(model.named_buffers())[name], before, rtol=0, atol=0)


@pytest.mark.parametrize('method', ['IBP', 'CROWN-IBP', 'SABR', 'TAPS'])
def test_zero_margins_are_not_reported_as_robust(method):
    data, target = torch.tensor([[.3], [.7]]), torch.tensor([0, 1])
    native = nn.Sequential(nn.Linear(1, 2)).eval()
    with torch.no_grad():
        native[0].weight.zero_()
        native[0].bias.zero_()
    model = BoundedModule(native, data, device='cpu')
    region = PerturbationLpNorm(norm=float('inf'), eps=.1, x_L=data-.1, x_U=data+.1)
    common = dict(hardened_model=model, data=data, target=target, n_classes=2,
                  criterion=nn.CrossEntropyLoss(), return_stats=True)
    if method == 'IBP':
        result = get_ibp_loss(ptb=region, **common)
    elif method == 'CROWN-IBP':
        result = get_crown_ibp_loss(ptb=region, beta=.5, **common)
    elif method == 'SABR':
        result = get_sabr_loss(original_model=native, eps=.1, subselection_ratio=.8,
                               data_min=0., data_max=1., pgd_steps=0, device='cpu', **common)
    else:
        with patch('CTRAIN.train.certified.losses.taps.bound_taps', return_value=(torch.zeros(2, 1), torch.zeros(2, 1))):
            result = get_taps_loss(original_model=native, bounded_blocks=[None, None],
                                   ptb=region, device='cpu', **common)
    assert result[-2 if method == 'SABR' else -1] == 1.


def test_checkpoint_refuses_overwrite_under_optimized_python(tmp_path):
    path = tmp_path / '1_checkpoint.pt'
    path.write_bytes(b'keep this checkpoint')
    code = f'''from CTRAIN.util import save_checkpoint
try:
    save_checkpoint(None, None, 0., 1, {str(tmp_path)!r}, overwrite=False)
except FileExistsError:
    pass
else:
    raise RuntimeError("overwrite protection vanished")
'''
    result = subprocess.run([sys.executable, '-O', '-c', code], cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert path.read_bytes() == b'keep this checkpoint'


@pytest.mark.parametrize('fail', [False, True])
def test_validation_loss_is_sample_weighted_and_restores_state(fail):
    from CTRAIN.train.certified.util import evaluate_validation_loss
    data = torch.tensor([[1.], [2.], [3.], [9.]])
    loader = DataLoader(TensorDataset(data, torch.zeros(4, dtype=torch.long)), batch_size=3)
    model = nn.Linear(1, 1).train()
    flags = [p.requires_grad for p in model.parameters()]
    def objective(x, y):
        assert not model.training
        if fail:
            next(model.parameters()).requires_grad_(False)
            raise RuntimeError('validation failed')
        return x.mean()
    if fail:
        with pytest.raises(RuntimeError, match='validation failed'):
            evaluate_validation_loss((model,), loader, objective, 'cpu')
    else:
        assert evaluate_validation_loss((model,), loader, objective, 'cpu') == 3.75
    assert model.training and [p.requires_grad for p in model.parameters()] == flags
