"""Run with PYTHONPATH=. python tests/test_taps_staps.py (requires auto_LiRPA)."""
import copy
import importlib
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from auto_LiRPA import BoundedModule, PerturbationLpNorm

from CTRAIN.bound.taps import split_bound_graph, bound_taps, _get_pivotal_points_one_batch, _get_bound_estimation_from_pts
from CTRAIN.bound.sabr import get_propagation_region
from CTRAIN.train.certified.losses.taps import get_taps_loss
from CTRAIN.train.certified.util import evaluate_taps_proxy
from CTRAIN.train.certified.progress import update_progress


def check():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    linear = nn.Linear(2, 3, bias=False).double()
    linear.weight.data.copy_(torch.tensor([[0., 0.], [1., -1.], [-2., 1.]]))
    lb, ub = torch.tensor([[-1., -2.]], dtype=torch.double), torch.tensor([[2., 3.]], dtype=torch.double)
    c = torch.tensor([[[1., -1., 0.], [1., 0., -1.]]], dtype=torch.double)
    linear.weight.grad = torch.ones_like(linear.weight)
    pts = _get_pivotal_points_one_batch(linear, lb, ub, 10, 2, None, .1, (), c, 3)
    assert pts.device == lb.device and pts.dtype == lb.dtype
    assert torch.allclose(_get_bound_estimation_from_pts(linear, pts, None, c), torch.tensor([[0., 4., 5.]], dtype=torch.double))
    assert torch.equal(linear.weight.grad, torch.ones_like(linear.weight))

    # Clipped, unequal-width and zero-width coordinates; both region APIs agree.
    x = torch.tensor([[0., .5, 1.]])
    low, high = (x - .1).clamp(0, 1), (x + .1).clamp(0, 1)
    high[0, 2] = low[0, 2]
    args = dict(model=nn.Identity(), data=x, target=torch.tensor([0]),
                subselection_ratio=.8, step_size=.5, n_steps=1, restarts=1)
    with patch('CTRAIN.bound.sabr.pgd_attack', side_effect=lambda **kw: kw['x_L']):
        mid, tau, _ = get_propagation_region(**args, x_L=low, x_U=high)
        assert torch.allclose(2 * tau, .8 * (high - low))
        assert (mid - tau >= low - 1e-7).all() and (mid + tau <= high + 1e-7).all()
        mid, tau, _ = get_propagation_region(**args, eps=.1, data_min=0., data_max=1.)
        assert torch.allclose(mid + tau, torch.tensor([[.08, .56, .98]]), atol=1e-6)

    data = torch.rand(4, 1, 2, 2)
    target = torch.tensor([0, 1, 0, 1])
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 3), nn.ReLU(), nn.BatchNorm1d(3), nn.Linear(3, 2))
    # Expose the split API without registering the layer alias twice.
    object.__setattr__(model, 'layers', model[:])
    model.eval()
    full = BoundedModule(model, data, device='cpu')
    blocks = split_bound_graph(full, (3, 2), model.layers)
    full.bounded_blocks = blocks
    for sizes in ((2, 2), (5, 0), (5,), (3, 2.0)):
        try:
            split_bound_graph(full, sizes, model.layers)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid split accepted: {sizes}')

    for mod in (model, full, *blocks):
        mod.train()
    ptb = PerturbationLpNorm(x_L=(data-.1).clamp(0,1), x_U=(data+.1).clamp(0,1))
    loss, = get_taps_loss(model, full, blocks, nn.CrossEntropyLoss(reduction='none'), data, target, 2, ptb, device='cpu', pgd_steps=2)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in full.parameters())
    optimizer = torch.optim.SGD(full.parameters(), lr=.01)
    before = [p.detach().clone() for p in full.parameters()]
    optimizer.step()
    assert any(not torch.equal(old, p) for old, p in zip(before, full.parameters()))

    for mod in (model, full, *blocks):
        mod.eval()
    buffers = [b.clone() for block in blocks for b in block.buffers()]
    bound_taps(model, full, blocks, data, target, 2, ptb, device='cpu', pgd_steps=2, pgd_step_size=None)
    assert all(not mod.training for mod in (model, full, *blocks))
    assert all(torch.equal(old, b) for old, b in zip(buffers, (b for block in blocks for b in block.buffers())))
    with patch('CTRAIN.bound.taps._get_pivotal_points', side_effect=RuntimeError('attack failed')):
        try:
            bound_taps(model, full, blocks, data, target, 2, ptb, device='cpu')
        except RuntimeError:
            pass
    assert all(not mod.training for mod in (model, full, *blocks))

    loader = DataLoader(TensorDataset(data, target), batch_size=4)
    loader.mean, loader.std = torch.tensor([0.]), torch.tensor([1.])
    loader.min, loader.max, loader.normalised = torch.tensor(0.), torch.tensor(1.), False
    for propagation in ('IBP', 'SABR'):
        region = None if propagation == 'IBP' else dict(subselection_ratio=.8, n_steps=2, early_stopping=False)
        accuracy = evaluate_taps_proxy(model, full, loader, .1, 2, 'cpu', propagation=propagation, sabr_args=region, pgd_steps=2, pgd_step_size=None)
        assert 0 <= accuracy <= 1
        assert all(not mod.training for mod in (model, full, *blocks))

    # Exercise both real training loops, selection, and warm-up logging.
    for name in ('taps', 'staps'):
        module = importlib.import_module(f'CTRAIN.train.certified.{name}')
        selected = []
        def validation(*args, **kwargs):
            selected.append(copy.deepcopy(full.state_dict()))
            return 1.0 if len(selected) == 1 else 0.0
        logs = []
        with patch.object(module, 'evaluate_taps_proxy', side_effect=validation), patch.object(module, 'log_epoch_summary', side_effect=lambda **kw: logs.append(kw)):
            getattr(module, f'{name}_train_model')(
                model, full, loader, val_loader=loader, num_epochs=10, end_epoch=3, eps=.1,
                eps_schedule=(0, 8), eps_scheduler_args={'start_eps': .1},
                optimizer=optimizer, n_classes=2, taps_pgd_steps=2,
                results_path=None, device='cpu', l1_regularisation_weight=None,
                **({'sabr_pgd_steps': 2} if name == 'staps' else {}),
            )
        assert len(selected) == 3
        assert all(torch.equal(full.state_dict()[key], value) for key, value in selected[0].items())
        with patch.object(module, 'log_epoch_summary', side_effect=lambda **kw: logs.append(kw)):
            getattr(module, f'{name}_train_model')(model, full, loader, num_epochs=9, end_epoch=1, eps=.1, eps_schedule=(1, 8), optimizer=optimizer, n_classes=2, results_path=None, device='cpu', l1_regularisation_weight=None)
        assert logs[-1]['robust_acc'] == logs[-1]['nat_acc']
        assert 'cert_acc' not in logs[-1]
    with patch('CTRAIN.train.certified.progress.tqdm') as bar:
        update_progress(bar, 1., robust_acc=.5)
        assert 'robust_proxy' in bar.set_postfix.call_args.args[0]
    print('TAPS/STAPS CPU regression checks passed')


if __name__ == '__main__':
    check()
