"""Runnable numerical and integration regressions for LiRPA-based set training."""
import copy
from itertools import product
import tempfile
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from CTRAIN.bound.set_based_zono import Zonotope, box_zonotope, bound_zonotope, zonotope_margins, sequential_layers
from CTRAIN.train.certified.losses.set_based import attack_zonotope, get_set_based_loss
from CTRAIN.train.certified.set_based import clean_bn_statistics, scheduled_epsilon
from CTRAIN.eval import eval_acc, eval_certified
from CTRAIN.util.util import preserve_model_state
from CTRAIN import SetBasedModelWrapper


def loader():
    data = torch.tensor([[.0, .8], [.4, 1.], [.5, .2], [.9, .7]], dtype=torch.double)
    result = DataLoader(TensorDataset(data, torch.tensor([0, 1, 0, 1])), batch_size=4)
    result.normalised = False
    result.min, result.max, result.std = torch.tensor(0.), torch.tensor(1.), torch.ones(2)
    return result


def numerical():
    model = nn.Sequential(nn.Linear(2, 2)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1., -2.], [1., -2.]], dtype=torch.double))
        model[0].bias.copy_(torch.tensor([1., 0.], dtype=torch.double))
    low = torch.tensor([[0., .3], [.2, .7]], dtype=torch.double)
    high = torch.tensor([[.4, .3], [.8, 1.]], dtype=torch.double)
    box = box_zonotope(low, high)
    out = bound_zonotope(model, box)
    torch.testing.assert_close(out.c, model((low + high)/2))
    torch.testing.assert_close(out.G, torch.einsum('bgf,kf->bgk', box.G, model[0].weight))
    torch.testing.assert_close(zonotope_margins(out, torch.zeros(2, dtype=torch.long))[:, 1], torch.ones(2, dtype=torch.double))
    relu_model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2)).double()
    for mode in ('interval', 'generators'):
        bound = bound_zonotope(relu_model, box, mode)
        lower, upper = bound.bounds()
        for _ in range(100):
            point = low + torch.rand_like(low) * (high - low)
            logits = relu_model(point)
            assert (logits >= lower - 1e-12).all() and (logits <= upper + 1e-12).all()
    # Explicit unstable, positive, negative, and zero-width ReLU coordinates.
    c = torch.tensor([[.2, 2., -2., 0.]], dtype=torch.double, requires_grad=True)
    g = torch.tensor([[[.8, .1, .2, 0.]]], dtype=torch.double, requires_grad=True)
    r = torch.tensor([[.1, .1, .1, 0.]], dtype=torch.double, requires_grad=True)
    relu = nn.Sequential(nn.ReLU())
    for mode in ('interval', 'generators'):
        def fn(c, g, r):
            out = bound_zonotope(relu, Zonotope(c, g, r), mode)
            return torch.cat((out.c.flatten(), out.G.flatten(), out.r.flatten()))
        # Gradcheck only away from stable/unstable and zero-width boundaries.
        assert torch.autograd.gradcheck(fn, (c[:, :3], g[:, :, :3], r[:, :3]))
        result = bound_zonotope(relu, Zonotope(c, g, r), mode)
        assert torch.isfinite(result.c).all()
    # Gradcheck affine weights includes BOTH center and correlated generator paths.
    weight = torch.tensor([[.4, -.7], [.9, .3]], dtype=torch.double, requires_grad=True)
    center = torch.tensor([[.2, .6]], dtype=torch.double)
    generator = torch.tensor([[[.1, .3], [.2, -.1]]], dtype=torch.double)
    def affine_objective(w):
        affine = nn.Linear(2, 2, bias=False).double()
        affine._parameters['weight'] = w
        z = bound_zonotope(nn.Sequential(affine), Zonotope(center, generator, torch.zeros_like(center)))
        return z.c.square().sum() + z.G.square().sum()
    assert torch.autograd.gradcheck(affine_objective, (weight,))
    affine_objective(weight).backward()
    expected = 2 * ((center @ weight.T).T @ center + (generator.flatten(0, 1) @ weight.T).T @ generator.flatten(0, 1))
    torch.testing.assert_close(weight.grad, expected)
    # Zero norm must have finite zero size gradients.
    zero = nn.Sequential(nn.Linear(2, 2, bias=False)).double()
    zero[0].weight.data.zero_()
    loss = get_set_based_loss(zero, low, torch.tensor([0, 1]), .1, .1, 0., 1., tau=1.)
    loss.backward()
    assert torch.equal(zero[0].weight.grad, torch.zeros_like(zero[0].weight))
    conv = nn.Sequential(nn.Conv2d(1, 2, 2), nn.Flatten(), nn.Linear(8, 2)).double()
    cb = box_zonotope(torch.zeros(1, 1, 3, 3, dtype=torch.double), torch.ones(1, 1, 3, 3, dtype=torch.double))
    co = bound_zonotope(conv, cb)
    torch.testing.assert_close(co.c, conv(cb.c))
    for _ in range(10):
        y = conv(torch.rand_like(cb.c)); l, u = co.bounds()
        assert (y >= l - 1e-12).all() and (y <= u + 1e-12).all()


def attacks_and_modes():
    model = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 2)).double()
    data, labels = next(iter(loader()))
    model.train(); model[1].eval()
    modes = [m.training for m in model.modules()]
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    list(model.parameters())[-1].requires_grad_(False)
    for factor in (.5, 1., 3.):
        z = attack_zonotope(model, data, labels, .2, 0., 1., 8, factor)
        lower, upper = z.bounds()
        assert (lower >= (data - .2).clamp(0, 1) - 1e-12).all()
        assert (upper <= (data + .2).clamp(0, 1) + 1e-12).all()
        assert not z.c.requires_grad and not z.G.requires_grad
    zero_width = attack_zonotope(model, data, labels, torch.tensor([.2, 0.], dtype=torch.double), 0., 1., 8, 3.)
    assert (zero_width.G[:, :, 1] == 0).all()
    assert not list(model.parameters())[-1].requires_grad
    assert modes == [m.training for m in model.modules()]
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    class Failing(nn.Sequential):
        def forward(self, x):
            raise RuntimeError('injected')
    broken = Failing(nn.Linear(2, 2)).double().train()
    try:
        attack_zonotope(broken, data, labels, .1, 0., 1.)
        assert False
    except RuntimeError:
        assert broken.training and broken[0].training
    class Residual(nn.Module):
        def __init__(self):
            super().__init__(); self.linear = nn.Linear(2, 2)
        def forward(self, x):
            return self.linear(x) + x
    for bad in (Residual(), nn.Sequential(nn.Sigmoid())):
        try:
            sequential_layers(bad); assert False
        except ValueError:
            pass


def training():
    from CTRAIN.model_wrappers.configs import get_config_space
    data_loader = loader()
    for input_set, error_propagation in product(('box', 'attack'), ('interval', 'generators')):
        model = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 2)).double()
        weights = copy.deepcopy(model.state_dict())
        results = []
        for chunk in (1, 4):
            clone = copy.deepcopy(model); clone.load_state_dict(weights)
            wrapper = SetBasedModelWrapper(clone, (2,), .1, 3, warm_up_epochs=1, ramp_up_epochs=1,
                                           device='cpu', input_set=input_set, propagation_batch_size=chunk,
                                           error_propagation=error_propagation,
                                           lr_decay_kwargs=dict(milestones=[2], gamma=.1))
            torch.manual_seed(7)
            wrapper.train_model(data_loader)
            updates = 1 + 2 * (len(data_loader.dataset) // chunk)
            assert clone[1].num_batches_tracked.item() == updates
            assert all(torch.isfinite(p).all() for p in clone.parameters())
            assert clone.training
            results.append(wrapper.state_dict())
            with preserve_model_state(clone):
                clean = eval_acc(clone, data_loader, device='cpu')
            cert, _ = eval_certified(clone, data_loader, .1, n_classes=2, method='FORWARD',
                                     propagation_batch_size=chunk, device='cpu')
            assert 0 <= cert <= clean <= 1
            assert clone[1].num_batches_tracked.item() == updates
        # Nominal statistics are chunk-local, so BN training is chunk-dependent.
        assert not torch.equal(results[0]['1.running_var'], results[1]['1.running_var'])
    with tempfile.TemporaryDirectory() as directory:
        def make():
            return SetBasedModelWrapper(nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2)).double(),
                                       (2,), .1, 3, device='cpu', warm_up_epochs=1, ramp_up_epochs=1,
                                       checkpoint_save_path=directory, tau=.2, num_generators=3,
                                       pgd_eps_factor=2., propagation_batch_size=2,
                                       lr_decay_kwargs=dict(milestones=[2], gamma=.1))
        initial = make(); state = copy.deepcopy(initial.state_dict())
        initial.train_model(data_loader, end_epoch=2)
        checkpoint = torch.load(Path(directory) / 'checkpoint_2.pt', weights_only=False)
        assert set(checkpoint) == {'model_state_dict', 'optimizer_state_dict',
                                   'lr_scheduler_state_dict', 'epoch', 'loss', 'selection_state'}
        assert all(path.suffix == '.pt' for path in Path(directory).iterdir())
        resumed = make()
        resumed.resume_from_checkpoint(Path(directory) / 'checkpoint_2.pt', data_loader)
        full = make(); full.load_state_dict(state); full.train_model(data_loader)
        for key, value in full.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[key])
        config = dict(get_config_space(full, 3, .1).get_default_configuration())
        recreated = full._recreate_for_hpo(config, 3)
        for key in ('input_set', 'num_generators', 'error_propagation', 'propagation_batch_size', 'pgd_eps_factor'):
            assert getattr(recreated, key) == getattr(full, key)
        full.train_model(data_loader, data_loader)
        assert full.selection_state['epoch'] >= 2
        # Evaluation caller modes survive certification and BN buffers do not change.
        full.eval(); eval_certified(full.original_model, data_loader, .1, n_classes=2, method='FORWARD', device='cpu')
        assert not full.original_model.training
    assert scheduled_epsilon(.1, 0, 0, 1, 1, 2) == 0
    assert scheduled_epsilon(.1, 1, 0, 1, 1, 2) == .05
    assert scheduled_epsilon(.1, 2, 0, 1, 1, 2) == .1


def global_gradient_clipping():
    from unittest.mock import patch
    data_loader = loader()
    for clip in (10., None):
        model = nn.Sequential(nn.Linear(2, 2)).double()
        wrapper = SetBasedModelWrapper(model, (2,), .1, 1, device='cpu',
                                       optimizer_func=torch.optim.SGD, gradient_clip=clip)
        before = copy.deepcopy(model.state_dict())
        # Known unequal gradient sizes distinguish global from per-tensor clipping.
        gradients = {'0.weight': torch.full_like(model[0].weight, 1000),
                     '0.bias': torch.ones_like(model[0].bias)}
        norm = torch.cat([value.flatten() for value in gradients.values()]).norm()
        scale = 1. if clip is None else min(1., clip / (norm.item() + 1e-6))
        with patch('CTRAIN.train.certified.set_based.get_set_based_loss',
                   side_effect=lambda *args, **kwargs: 1000 * model[0].weight.sum() + model[0].bias.sum()):
            wrapper.train_model(data_loader)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key] - wrapper.lr * scale * gradients[key])



def convolutional_bn_and_presets():
    import json
    from unittest.mock import patch
    from CTRAIN.model_definitions import CNN6_SetBased
    model = nn.Sequential(nn.Conv2d(1, 2, 2), nn.BatchNorm2d(2), nn.ReLU(),
                          nn.Flatten(), nn.Linear(8, 2), nn.BatchNorm1d(2)).double()
    data = torch.rand(4, 1, 3, 3, dtype=torch.double)
    reference = copy.deepcopy(model)
    with torch.no_grad():
        expected = reference(data)
    stats = clean_bn_statistics(model, data, sequential_layers(model))
    point = Zonotope(data, data.new_empty((4, 0, 1, 3, 3)), torch.zeros_like(data))
    result = bound_zonotope(model, point, bn_statistics=stats)
    torch.testing.assert_close(result.c, expected)
    assert all(layer.num_batches_tracked == 1 for layer in model.modules()
               if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)))
    model.eval()
    torch.testing.assert_close(bound_zonotope(model, point).c, model(data))
    for options in ({'tau': -1}, {'pgd_eps_factor': float('nan')}, {'num_generators': 0},
                    {'propagation_batch_size': 0}, {'input_set': 'bad'}, {'error_propagation': 'drop'}):
        try:
            SetBasedModelWrapper(nn.Sequential(nn.Linear(2, 2)), (2,), .1, 3, device='cpu', **options)
            assert False
        except ValueError:
            pass
    wrapper = SetBasedModelWrapper(nn.Sequential(nn.Linear(2, 2)).double(), (2,), .1, 3,
                                   warm_up_epochs=0, ramp_up_epochs=0, device='cpu', population_bn=True)
    with patch('CTRAIN.train.certified.set_based.eval_certified', return_value=(.5, {})):
        wrapper.train_model(loader(), loader())
    assert wrapper.selection_state['epoch'] == 1
    for key, value in wrapper.state_dict().items():
        torch.testing.assert_close(value, wrapper.selection_state['weights'][key])
    with patch('CTRAIN.eval.eval.eval_adversarial', return_value=.5):
        result = wrapper.evaluate(loader())
        assert len(result) == 3 and result[2] == .5
    # All evaluation modes and existing parameter gradients are restored.
    assert wrapper.original_model.training
    bn_wrapper = SetBasedModelWrapper(copy.deepcopy(model), (1, 3, 3), .1, 1, device='cpu', population_bn=True)
    images = DataLoader(TensorDataset(data, torch.tensor([0, 1, 0, 1])), batch_size=4)
    images.normalised = False; images.min = torch.tensor(0.); images.max = torch.tensor(1.)
    bn_wrapper.train_model(images)
    for layer in bn_wrapper.original_model.modules():
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            assert layer.momentum == .1 and layer.num_batches_tracked.item() == 3



def actual_evaluation():
    from CTRAIN.model_wrappers.model_wrapper import CTRAINWrapper
    images = torch.rand(2, 1, 2, 2)
    data_loader = DataLoader(TensorDataset(images, torch.tensor([0, 1])), batch_size=2)
    data_loader.min, data_loader.max, data_loader.std = torch.tensor(0.), torch.tensor(1.), torch.ones(1)
    data_loader.normalised = False
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    wrapper = SetBasedModelWrapper(model, (1, 2, 2), .1, 1, device='cpu')
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    result = wrapper.evaluate(data_loader, test_samples=1)
    assert len(result) == 3 and all(0 <= value <= 1 for value in result)
    assert model.training
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    result = wrapper.evaluate(data_loader, test_samples=1, eval_method='IBP')
    assert len(result) == 3
    # Existing wrappers expose the same forward evaluation choice.
    existing = CTRAINWrapper(copy.deepcopy(model), .1, (1, 1, 2, 2), device='cpu')
    result = existing.evaluate(data_loader, test_samples=1, eval_method='FORWARD')
    assert len(result) == 3



def lirpa_operator_regressions():
    from contextlib import ExitStack
    from unittest.mock import patch
    from auto_LiRPA.operators import (BoundLinear, BoundConv, BoundBatchNormalization,
                                     BoundRelu, BoundFlatten, BoundIdentity)
    network = nn.Sequential(nn.Conv2d(1, 2, 2), nn.BatchNorm2d(2), nn.ReLU(),
                            nn.Identity(), nn.Flatten(), nn.Linear(8, 2)).double().eval()
    data = torch.rand(2, 1, 3, 3, dtype=torch.double)
    operators = [BoundLinear, BoundConv, BoundBatchNormalization,
                 BoundRelu, BoundFlatten, BoundIdentity]
    with ExitStack() as stack:
        calls = [stack.enter_context(patch.object(cls, 'bound_forward', autospec=True,
                                                  side_effect=cls.bound_forward)) for cls in operators]
        bound_zonotope(network, box_zonotope(data - .1, data + .1))
        assert all(call.call_count for call in calls), 'All supported operations must use LiRPA'
    try:
        bound_zonotope(network, box_zonotope(data, data), layers=[nn.Sigmoid()])
        assert False
    except ValueError:
        pass
    # PyTorch's string padding and non-affine BN remain supported by the adapter.
    padded = nn.Sequential(nn.Conv2d(1, 2, 3, padding='same'),
                           nn.BatchNorm2d(2, affine=False)).double().eval()
    point = Zonotope(data, data.new_empty((2, 0, 1, 3, 3)), torch.zeros_like(data))
    torch.testing.assert_close(bound_zonotope(padded, point).c, padded(data))
    # Zero-weight center/generator gradients must remain the exact affine gradients.
    for layer in [nn.Linear(2, 2), nn.Conv2d(1, 1, 1), nn.BatchNorm1d(2)]:
        layer = layer.double().eval()
        layer.weight.data.zero_()
        shape = (2, 1, 2, 2) if isinstance(layer, nn.Conv2d) else (2, 2)
        data = torch.rand(shape, dtype=torch.double)
        if isinstance(layer, nn.BatchNorm1d):
            layer.running_mean.fill_(.3)
        center = data.clone().requires_grad_(True)
        generators = torch.full((2, 1, *shape[1:]), .1, dtype=torch.double)
        enclosure = Zonotope(center, generators, torch.full_like(data, .05))
        output = bound_zonotope(nn.Sequential(layer), enclosure)
        point_grads = torch.autograd.grad(layer(center).sum(), tuple(layer.parameters()), retain_graph=True)
        set_grads = torch.autograd.grad(output.c.sum(), tuple(layer.parameters()), retain_graph=True)
        for actual, expected in zip(set_grads, point_grads):
            torch.testing.assert_close(actual, expected)
        generator_grads = torch.autograd.grad(output.G.sum(), layer.weight)[0]
        expected = torch.full_like(layer.weight, .2 if isinstance(layer, (nn.Linear, nn.BatchNorm1d)) else .8)
        if isinstance(layer, nn.BatchNorm1d):
            expected /= torch.sqrt(layer.running_var + layer.eps)
        torch.testing.assert_close(generator_grads, expected)
    # LiRPA's tiny-width clamp must not change the specified ReLU enclosure.
    for dtype in (torch.float32, torch.float64):
        c = torch.tensor([[1e-12]], dtype=dtype, requires_grad=True)
        g = torch.tensor([[[3e-12]]], dtype=dtype, requires_grad=True)
        for mode in ('interval', 'generators'):
            out = bound_zonotope(nn.Sequential(nn.ReLU()), Zonotope(c, g, torch.zeros_like(c)), mode)
            torch.testing.assert_close(out.c, torch.full_like(c, 4e-12/3), atol=0, rtol=1e-5)
            torch.testing.assert_close(out.G[:, :1], torch.full_like(g, 2e-12), atol=0, rtol=1e-5)
            error = out.r if mode == 'interval' else out.G[:, -1]
            torch.testing.assert_close(error, torch.full_like(c, 2e-12/3), atol=0, rtol=1e-5)
            gradients = torch.autograd.grad(out.c.sum() + out.G.sum() + out.r.sum(), (c, g), retain_graph=True)
            assert all(torch.isfinite(gradient).all() for gradient in gradients)


def main():
    torch.set_num_threads(1); torch.manual_seed(42)
    numerical(); attacks_and_modes(); training(); global_gradient_clipping(); convolutional_bn_and_presets(); actual_evaluation(); lirpa_operator_regressions()
    print('Set-based numerical and integration checks passed')


if __name__ == '__main__':
    main()
