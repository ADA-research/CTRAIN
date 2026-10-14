"""Runnable numerical regressions for the general-property workflow."""
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from CTRAIN import PropertyModelWrapper
from CTRAIN.properties import Outputs, OutputIndex, And, Or, Ge, Lt, Box, PropertySpec, PropertyGraph
from CTRAIN.bound.properties import PropertyBounds, box_endpoints
from CTRAIN.eval.properties import initial_partition, evaluate_cells, refine_partition, find_counterexample, validate_partition


def raises(error, function):
    try:
        function()
    except error:
        return
    raise AssertionError(f'Expected {error.__name__}')


def syntax():
    y = Outputs()
    values = torch.tensor([[0., 1., 2.], [.5, .2, 3.]], dtype=torch.float64)
    formula = And(y[0] >= 0, Or(y[1] < .5, y[2] >= 2))
    assert formula.evaluate(values).all()
    assert (formula.margin(values) >= 0).all()
    assert not Lt(OutputIndex(y, 0), 0).evaluate(values).any()
    assert Ge(y[0], 0).evaluate(values).all()
    raises(TypeError, lambda: bool(formula))
    raises(TypeError, lambda: y[0] == 0)
    raises(ValueError, lambda: y[-1])
    raises(ValueError, lambda: y[0] * y[1])
    raises(ValueError, lambda: Ge(y[0], 0, scale=0))
    raises(ValueError, lambda: Box([1], [0]))
    model = nn.Linear(3, 3, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.eye(3, dtype=torch.double))
    torch.testing.assert_close(PropertyGraph(model, formula)(values),
                               PropertyGraph(model, formula, relu=True)(values))


def bounds_and_refinement():
    y = Outputs()
    for dtype in (torch.float32, torch.float64):
        model = nn.Linear(1, 2).to(dtype)
        with torch.no_grad():
            model.weight.fill_(1)
            model.bias.copy_(torch.tensor([.25, 0.], dtype=dtype))
        spec = PropertySpec('cancel', And(y[0] - y[1] >= .1, y[1] >= -2), [Box([-1], [1])])
        bound = PropertyBounds(model, (1,), [spec])
        for method in ('IBP', 'CROWN-IBP', 'CROWN'):
            lower = bound(0, spec.domains, method)
            torch.testing.assert_close(lower, torch.tensor([[.15, 1.]], dtype=dtype))
            model.zero_grad(set_to_none=True)
            (-lower.sum()).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        point = Box([.4], [.4])
        lb = bound(0, [point], 'CROWN')
        assert (lb > 0).all()
        low, high = box_endpoints([Box([.1], [.3])], model.weight)
        assert low.double().item() <= .1 and high.double().item() >= .3
        composite = PropertySpec('or', Or(y[1] <= .2, y[1] >= -.2), [Box([-1], [1])])
        checker = PropertyBounds(model, (1,), [composite])
        cells = initial_partition([composite])
        rows, _ = evaluate_cells(checker, cells, 'IBP')
        assert rows[0]['outcome'] == 'unknown'
        children = refine_partition(cells, rows, [composite])
        validate_partition(children, [composite])
        raises(ValueError, lambda: validate_partition(children[:1], [composite]))
        assert len(children) == 2 and sum(c['probability'] for c in children) == 1
        assert torch.equal(children[0]['box'].lower, cells[0]['box'].lower)
        assert torch.equal(children[1]['box'].upper, cells[0]['box'].upper)
        assert torch.equal(children[0]['box'].upper, children[1]['box'].lower)
        rows, _ = evaluate_cells(checker, children, 'IBP')
        assert all(r['outcome'] == 'certified' for r in rows)
        samples = torch.linspace(-1, 1, 101, dtype=dtype).unsqueeze(1)
        # Certification must optimize slopes even under no_grad, preserving
        # existing weight gradients and requires_grad flags.
        alpha_model = nn.Sequential(nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 2)).to(dtype)
        alpha_checker = PropertyBounds(alpha_model, (1,), [spec, composite])
        for parameter in alpha_model.parameters():
            parameter.grad = torch.ones_like(parameter)
        flags = [p.requires_grad for p in alpha_model.parameters()]
        for index, alpha_spec in enumerate((spec, composite)):
            with torch.no_grad():
                alpha_lb = alpha_checker(index, alpha_spec.domains, 'ALPHA-CROWN')
                margins = alpha_spec.formula.margin(alpha_model(samples))
            assert torch.isfinite(alpha_lb).all()
            assert (margins >= alpha_lb.min() - 1e-6).all()
        assert flags == [p.requires_grad for p in alpha_model.parameters()]
        assert all(torch.equal(p.grad, torch.ones_like(p)) for p in alpha_model.parameters())
        for method in ('IBP', 'CROWN-IBP', 'CROWN'):
            lb = checker(0, composite.domains, method)
            assert (composite.formula.margin(model(samples)) >= lb - 1e-6).all()
            model.zero_grad(set_to_none=True)
            (-lb.mean()).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        modes = [m.training for m in model.modules()]
        with patch.object(bound.raw, 'compute_bounds', side_effect=RuntimeError('injected')):
            raises(RuntimeError, lambda: bound(0, spec.domains))
        assert modes == [m.training for m in model.modules()]
        old = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
        witness = find_counterexample(model, y[1] >= .5, Box([-1], [1]))
        assert witness is not None
        for p, grad in zip(model.parameters(), old):
            if grad is not None:
                torch.testing.assert_close(p.grad, grad)


def training_and_resume():
    y = Outputs()
    spec = PropertySpec('positive', y[0] >= 0, [Box([-1], [1])])
    data = torch.tensor([[-1.], [0.], [1.], [.5]])
    loader = DataLoader(TensorDataset(data, torch.ones_like(data)), batch_size=2)
    torch.manual_seed(4)
    initial = nn.Linear(1, 1)
    with torch.no_grad():
        initial.weight.zero_(); initial.bias.fill_(.1)
    def make():
        return PropertyModelWrapper(copy.deepcopy(initial), (1,), properties=[spec],
                                    task_loss=nn.MSELoss(), num_epochs=2, property_batch_size=2)
    full = make()
    rng = torch.get_rng_state()
    original = copy.deepcopy(full.original_model.state_dict())
    full.train_model(loader)
    assert any(not torch.equal(original[k], v) for k, v in full.original_model.state_dict().items())
    assert full.certify_properties(refine=False)['certified']
    with tempfile.TemporaryDirectory() as directory:
        split = make(); torch.set_rng_state(rng)
        split.train_model(loader, end_epoch=1)
        filename = Path(directory) / 'checkpoint.pt'
        split.save_checkpoint(filename, loader)
        resumed = make().load_checkpoint(filename)
        resumed.train_model(loader)
        for name, value in full.original_model.state_dict().items():
            torch.testing.assert_close(value, resumed.original_model.state_dict()[name], rtol=0, atol=0)
        bad = PropertyModelWrapper(copy.deepcopy(initial), (1,), properties=[spec],
            task_loss=nn.MSELoss(), num_epochs=2, property_batch_size=2, margin=.1)
        raises(ValueError, lambda: bad.load_checkpoint(filename))
        selected = make()
        selected.train_model(loader, val_loader=loader, end_epoch=1)
        raises(ValueError, lambda: selected.save_checkpoint(Path(directory) / 'selected.pt'))
        raises(ValueError, lambda: selected.train_model(loader))
        selected.load_checkpoint(filename).train_model(loader)
    bn_model = nn.Sequential(nn.Linear(1, 2), nn.BatchNorm1d(2), nn.ReLU(), nn.Linear(2, 1))
    composite = PropertySpec('bn', Or(y[0] >= -10, y[0] <= 10), [Box([-1], [1])])
    bn = PropertyModelWrapper(bn_model, (1,), properties=[composite], task_loss=nn.MSELoss(),
        num_epochs=1, property_batch_size=2, population_bn=True)
    bn_model.train(); momentum = bn_model[1].momentum
    bn.train_model(loader)
    assert bn_model.training and bn_model[1].momentum == momentum
    assert bn_model[1].num_batches_tracked > 0
    result = bn.certify_properties(refine=False)
    assert result['certified'] and bn_model.training
    buffers = dict(bn.verification_model.named_buffers())
    for name, buf in bn.bounds.raw.named_buffers():
        torch.testing.assert_close(buf, buffers[name])


def shared_constant_joint_gradients():
    class NormalizedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('offset', torch.tensor([.3, -.2]))
            self.register_buffer('scale', torch.tensor([.7, 1.3]))
            self.layers = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2))

        def forward(self, inputs):
            return self.layers((inputs - self.offset) / self.scale)

    from CTRAIN.train.certified.losses.properties import property_loss
    torch.manual_seed(7)
    model = NormalizedModel()
    y = Outputs()
    specs = [PropertySpec('atomic', y[0] >= 10, [Box([0, 0], [1, 1])]),
             PropertySpec('composite', Or(y[0] >= 10, y[1] >= 10), [Box([1, 1], [2, 2])]),
             PropertySpec('conjunction', And(y[0] >= 10, y[1] >= 10), [Box([-1, -1], [0, 0])])]
    bounds = PropertyBounds(model, (2,), specs)
    parameters = tuple(model.parameters())
    for method in ('IBP', 'CROWN-IBP', 'CROWN'):
        separate = [torch.zeros_like(parameter) for parameter in parameters]
        for index, spec in enumerate(specs):
            loss = property_loss(bounds(index, list(spec.domains) * 3, method))
            assert loss > 0
            for total, gradient in zip(separate, torch.autograd.grad(loss, parameters)):
                total.add_(gradient)
        losses = [property_loss(bounds(index, list(spec.domains) * 3, method))
                  for index, spec in enumerate(specs)]
        for actual, expected in zip(torch.autograd.grad(sum(losses), parameters), separate):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    wrapper = PropertyModelWrapper(model, (2,), properties=specs, task_loss=nn.MSELoss(),
        num_epochs=1, bound_method='CROWN-IBP', property_batch_size=6)
    data = torch.tensor([[.5, .5], [1.5, 1.5], [-.5, -.5]])
    target = model(data).detach()
    before = copy.deepcopy(model.state_dict())
    with patch('CTRAIN.train.certified.properties.torch.multinomial',
               return_value=torch.tensor([0, 1, 2, 0, 1, 2])):
        wrapper.train_model(DataLoader(TensorDataset(data, target), batch_size=3))
    assert wrapper.history[0]['property_loss'] > 0
    assert any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters())


def main():
    torch.set_num_threads(1)
    syntax(); bounds_and_refinement(); training_and_resume(); shared_constant_joint_gradients()
    print('Property DSL, bounds, refinement, gradients, training, BN, and resume checks passed')


if __name__ == '__main__':
    main()
