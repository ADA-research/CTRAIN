"""Full-box LiRPA certification: sound margins, BN isolation, and fresh bounds."""
from unittest.mock import patch
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from auto_LiRPA import BoundedModule
from CTRAIN.eval import eval_forward, eval_certified
from CTRAIN import SetBasedModelWrapper
from CTRAIN.util.util import preserve_model_state
from CTRAIN.bound.set_based_zono import box_zonotope, bound_zonotope, zonotope_margins


def check(model, data, eps):
    labels = torch.arange(len(data)) % 2
    loader = DataLoader(TensorDataset(data, labels), batch_size=3)
    loader.min, loader.max, loader.normalised = 0., 1., False
    model.train()
    modes = [m.training for m in model.modules()]
    for index, parameter in enumerate(model.parameters()):
        parameter.requires_grad_(index % 2 == 0)
        parameter.grad = torch.ones_like(parameter)
    flags = [p.requires_grad for p in model.parameters()]
    options = copy.deepcopy(model.bound_opts) if isinstance(model, BoundedModule) else None
    relaxations = {node: node.relu_options for node in model.modules() if hasattr(node, 'relu_options')}
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    captured = []
    compute = BoundedModule.compute_bounds

    def record(self, *args, **kwargs):
        assert kwargs['method'] == 'Forward'
        assert all(node.relu_options == 'same-slope' for node in self.modules() if hasattr(node, 'relu_options'))
        chunk = kwargs['x'][0]
        torch.testing.assert_close(chunk.ptb.x_L, (chunk - eps).clamp(0, 1))
        torch.testing.assert_close(chunk.ptb.x_U, (chunk + eps).clamp(0, 1))
        result = compute(self, *args, **kwargs)
        captured.append((kwargs['x'][0].detach().clone(), kwargs['C'], result[0]))
        return result

    with patch.object(BoundedModule, 'compute_bounds', record):
        result, records = eval_certified(model, loader, eps, n_classes=None, method='FORWARD',
                                         propagation_batch_size=2, device='cpu')
    assert len(captured) == 3  # chunks 2+1, then 1; different batch shapes
    assert modes == [m.training for m in model.modules()]
    assert flags == [p.requires_grad for p in model.parameters()]
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    assert all(node.relu_options == option for node, option in relaxations.items())
    if options is not None:
        assert model.bound_opts == options
    assert list(records) == list(range(len(data)))
    assert all(record['method'] == 'FORWARD' and record['running_time'] >= 0 for record in records.values())
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    with preserve_model_state(model), torch.no_grad():
        model.eval()
        for chunk, C, margins in captured:
            lower, upper = (chunk - eps).clamp(0, 1), (chunk + eps).clamp(0, 1)
            for _ in range(20):
                point = lower + torch.rand_like(lower) * (upper - lower)
                actual = torch.bmm(C, model(point).unsqueeze(-1)).squeeze(-1)
                assert (actual >= margins - 1e-9).all()
        assert result == eval_certified(model, loader, eps, n_classes=2, method='FORWARD',
                                        propagation_batch_size=1, device='cpu')[0]
        count, total, records = eval_forward(model, eps, loader, 2, 3, 'cpu', 2)
        assert total == 3 and list(records) == [0, 1, 2]
        assert count == sum(record['result'] == 'unsat' for record in records.values())
        # Ordinary point bounds at zero radius agree with the clean classifier.
        clean = (model(data).argmax(1) == labels).double().mean().item()
        certified, _ = eval_certified(model, loader, 0., n_classes=2, method='FORWARD', device='cpu')
        assert clean == certified
    model.train()
    with patch.object(BoundedModule, 'compute_bounds', side_effect=RuntimeError('injected')):
        try:
            eval_certified(model, loader, eps, n_classes=2, method='FORWARD', device='cpu')
        except RuntimeError as error:
            assert str(error) == 'injected'
        else:
            raise AssertionError('Expected injected failure')
    assert all(m.training for m in model.modules())
    assert flags == [p.requires_grad for p in model.parameters()]
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    assert all(node.relu_options == option for node, option in relaxations.items())
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)


def wrapper_geometry_and_dispatch():
    data = torch.tensor([[[[0.]], [[1.]]], [[[.4]], [[.7]]]], dtype=torch.double)
    loader = DataLoader(TensorDataset(data, torch.tensor([0, 1])), batch_size=2)
    loader.normalised, loader.std = True, torch.tensor([.2, .4])
    loader.min, loader.max = torch.zeros(2, 1, 1), torch.ones(2, 1, 1)
    model = nn.Sequential(nn.BatchNorm2d(2), nn.Flatten(), nn.Linear(2, 2)).double()
    wrapper = SetBasedModelWrapper(model, (2, 1, 1), .1, 1, device='cpu',
                                   error_propagation='generators', propagation_batch_size=2)
    model[0].running_mean.fill_(.3)
    model[0].running_var.fill_(.7)
    saved = copy.deepcopy(model.state_dict())
    compute = BoundedModule.compute_bounds
    radius = torch.tensor([.5, .25], dtype=torch.double).reshape(2, 1, 1)

    def record(self, *args, **kwargs):
        chunk = kwargs['x'][0]
        torch.testing.assert_close(chunk.ptb.x_L, (chunk - radius).clamp(loader.min, loader.max))
        torch.testing.assert_close(chunk.ptb.x_U, (chunk + radius).clamp(loader.min, loader.max))
        for name, value in self.named_buffers():
            if name in saved:
                torch.testing.assert_close(value, saved[name])
        return compute(self, *args, **kwargs)

    with patch.object(BoundedModule, 'compute_bounds', record), \
            patch('CTRAIN.eval.eval.eval_adversarial', return_value=.5) as attack, \
            patch('CTRAIN.bound.set_based_zono.bound_zonotope', side_effect=AssertionError('training-only')):
        metrics = wrapper.evaluate(loader)
        assert len(metrics) == 3 and metrics[2] == .5
        torch.testing.assert_close(attack.call_args.kwargs['eps'], radius)
        assert attack.call_args.kwargs['restarts'] == 30
        assert attack.call_args.kwargs['n_steps'] == 100
        assert wrapper.evaluate(loader, eval_method='ZONOTOPE') == metrics
    assert model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name])


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    for eps in (0., .2):
        for bounded in (False, True):
            for model, data in [
                (nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 2)).double(),
                 torch.tensor([[0., 1.], [.3, .4], [.8, .2], [1., 0.]], dtype=torch.double)),
                (nn.Sequential(nn.Conv2d(1, 2, 2), nn.BatchNorm2d(2), nn.ReLU(),
                               nn.Flatten(), nn.Linear(8, 2)).double(), torch.rand(4, 1, 3, 3, dtype=torch.double))]:
                if bounded:
                    model.eval()
                    model = BoundedModule(model, data[:1], device='cpu')
                check(model, data, eps)
    # Shared final rows cancel when C is merged; output-box errors cannot cancel.
    model = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 2)).double().eval()
    with torch.no_grad():
        model[0].weight.fill_(1); model[0].bias.fill_(-.5)
        model[2].weight.fill_(1); model[2].bias.copy_(torch.tensor([.1, 0.]))
    data = torch.tensor([[.5]], dtype=torch.double)
    loader = DataLoader(TensorDataset(data, torch.tensor([0])))
    loader.min, loader.max = 0., 1.
    old = bound_zonotope(model, box_zonotope(data - .5, data + .5))
    assert zonotope_margins(old, torch.tensor([0]))[0, 1] < 0
    assert eval_certified(model, loader, .5, n_classes=2, method='FORWARD', device='cpu')[0] == 1.
    # IBP loses the correlation between two identical hidden features; forward
    # recovers it for the still-pending sample in a mixed-method evaluation.
    affine = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 2)).double().eval()
    with torch.no_grad():
        affine[0].weight.fill_(1); affine[0].bias.zero_()
        affine[1].weight.copy_(torch.tensor([[1., -1.], [0., 0.]]))
        affine[1].bias.copy_(torch.tensor([.1, 0.]))
    bounded = BoundedModule(affine, data, device='cpu')
    score, records = eval_certified(bounded, loader, .5, n_classes=2,
                                    method=['IBP', 'FORWARD'], device='cpu')
    assert score == 1. and records[0]['method'] == 'FORWARD'
    for options in ({'eps': -.1}, {'eps': float('nan')}, {'propagation_batch_size': 0},
                    {'test_samples': 0}, {'test_samples': 1.5}):
        try:
            eval_forward(model, data_loader=loader, n_classes=2, device='cpu', **{'eps': .1, **options})
        except ValueError:
            pass
        else:
            raise AssertionError(f'Expected invalid evaluation options: {options}')
    empty = DataLoader(TensorDataset(data[:0], torch.empty(0, dtype=torch.long)))
    try:
        eval_forward(model, .1, empty, device='cpu')
    except ValueError as error:
        assert 'contain samples' in str(error)
    else:
        raise AssertionError('Expected empty-loader rejection')
    wrapper_geometry_and_dispatch()
    print('LiRPA full-box certification checks passed')


if __name__ == '__main__':
    main()
