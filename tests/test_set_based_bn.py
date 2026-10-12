"""Set-based nominal BN gradients, statistics, and attack graph preservation."""
import copy
import math
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from CTRAIN.bound.zonotope import sequential_layers
from CTRAIN.model_wrappers import SetBasedModelWrapper
from CTRAIN.train.certified.set_based import clean_bn_statistics
from CTRAIN.train.certified.losses.set_based import attack_zonotope, get_set_based_loss


def numeric_and_robust(device):
    torch.manual_seed(16)
    model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.ReLU(), nn.Linear(3, 2)).double().to(device)
    reference = copy.deepcopy(model)
    data = torch.tensor([[.2, .3], [.4, .8], [.7, .6], [.8, .2]], dtype=torch.double, device=device)
    labels = torch.tensor([0, 1, 0, 1], device=device)
    expected = F.cross_entropy(reference(data), labels)
    expected.backward()
    stats = clean_bn_statistics(model, data, sequential_layers(model))
    actual = get_set_based_loss(model, data, labels, 0., 0., 0., 1., bn_statistics=stats)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    for parameter, want in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, want.grad, atol=1e-10, rtol=1e-8)
    assert model[1].num_batches_tracked == 1
    model.zero_grad(set_to_none=True)
    stats = clean_bn_statistics(model, data, sequential_layers(model))
    loss = get_set_based_loss(model, data, labels, .05, .05, 0., 1., bn_statistics=stats)
    loss.backward()
    assert model[0].bias.grad.abs().max() < 1e-10, 'Recomputed nominal BN must cancel a pre-BN shift'
    weight = model[0].weight.detach().clone().requires_grad_(True)
    def objective(value):
        model[0]._parameters['weight'] = value
        stats = clean_bn_statistics(model, data, sequential_layers(model))
        return get_set_based_loss(model, data, labels, .05, .05, 0., 1., bn_statistics=stats)
    assert torch.autograd.gradcheck(objective, (weight,), atol=1e-5, rtol=1e-4)



def statistics_and_training_step(device):
    for shape, bn in (((1, 2), nn.BatchNorm1d(2)), ((3, 2, 2, 2), nn.BatchNorm2d(2))):
        model = nn.Sequential(bn).double().to(device)
        data = torch.arange(math.prod(shape), dtype=torch.double, device=device).reshape(shape) / 10
        dims = (0, *range(2, len(shape)))
        mean, variance = data.mean(dims), data.var(dims, unbiased=False)
        stats = clean_bn_statistics(model, data, sequential_layers(model))
        torch.testing.assert_close(stats[bn][0], mean)
        torch.testing.assert_close(stats[bn][1], variance)
        torch.testing.assert_close(bn.running_mean, .01 * mean)
        torch.testing.assert_close(bn.running_var, .99 + .01 * variance)
        assert bn.num_batches_tracked == 1 and bn.momentum == .1
        model.eval()
        saved = copy.deepcopy(model.state_dict())
        clean_bn_statistics(model, data, sequential_layers(model))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[key])
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.ReLU(), nn.Linear(3, 2)).double().to(device)
    reference = copy.deepcopy(model)
    data = torch.tensor([[.2, .4], [.3, .8], [.7, .6]], dtype=torch.double, device=device)
    labels = torch.tensor([0, 1, 0], device=device)
    optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
    F.cross_entropy(reference(data), labels).backward()
    optimizer.step()
    loader = DataLoader(TensorDataset(data, labels), batch_size=3)
    loader.min, loader.max, loader.normalised = 0., 1., False
    wrapper = SetBasedModelWrapper(model, (2,), .1, 1, device=device,
                                  optimizer_func=torch.optim.SGD, lr=.01, gradient_clip=None)
    wrapper.train_model(loader)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)



def attack_statistics(device):
    torch.manual_seed(9)
    model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.ReLU(), nn.Linear(3, 2)).double().to(device)
    data = torch.tensor([[.2, .4], [.3, .8], [.7, .6]], dtype=torch.double, device=device)
    labels = torch.tensor([0, 1, 0], device=device)
    layers = sequential_layers(model)
    stats = clean_bn_statistics(model, data, layers)
    assert all(value.requires_grad for pair in stats.values() for value in pair)
    saved = copy.deepcopy(model.state_dict())
    modes = [module.training for module in model.modules()]
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    count, radius = 3, .1
    lower, upper = (data - radius).clamp(0, 1), (data + radius).clamp(0, 1)
    torch.manual_seed(19)
    points = torch.cat([lower + torch.rand_like(data) * (upper - lower) for _ in range(count)]).requires_grad_(True)
    logits = points
    for layer in layers:
        if isinstance(layer, nn.BatchNorm1d):
            mean, variance = stats[layer]
            logits = F.batch_norm(logits, mean.detach(), variance.detach(), layer.weight, layer.bias,
                                  training=False, eps=layer.eps)
        else:
            logits = layer(logits)
    gradient, = torch.autograd.grad(F.cross_entropy(logits, labels.repeat(count)), points)
    positions = (points + radius * gradient.sign()).clamp(lower.repeat(count, 1), upper.repeat(count, 1)).detach()
    positions = positions.reshape(count, *data.shape).transpose(0, 1)
    center = positions.mean(1)
    generators = (positions - center[:, None]) / count
    extent = generators.abs().sum(1)
    room = torch.minimum(center - lower, upper - center).clamp_min(0)
    generators *= (room / torch.where(extent > 0, extent, torch.ones_like(extent))).clamp(max=1)[:, None]
    torch.manual_seed(19)
    actual = attack_zonotope(model, data, labels, radius, 0., 1., count, bn_statistics=stats, layers=layers)
    torch.testing.assert_close(actual.c, center)
    torch.testing.assert_close(actual.G, generators)
    assert modes == [module.training for module in model.modules()]
    assert all(torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[key])
    with patch('CTRAIN.train.certified.losses.set_based.bound_zonotope', side_effect=RuntimeError('injected')):
        try:
            attack_zonotope(model, data, labels, radius, 0., 1., count, bn_statistics=stats, layers=layers)
            raise AssertionError('Expected attack failure')
        except RuntimeError:
            pass
    assert modes == [module.training for module in model.modules()]
    model.zero_grad(set_to_none=True)
    # The attack must leave the nominal graph usable by the robust training loss.
    loss = get_set_based_loss(model, data, labels, radius, radius, 0., 1., input_set='attack',
                              bn_statistics=stats, layers=layers)
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert model[0].bias.grad.abs().max() < 1e-10



if __name__ == "__main__":
    torch.set_num_threads(1)
    numeric_and_robust("cpu")
    statistics_and_training_step("cpu")
    attack_statistics("cpu")
    print("Set-based nominal BN regressions passed")
