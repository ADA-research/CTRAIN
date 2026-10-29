"""Check bounds against exact affine extrema and enumerated ReLU outputs."""
import itertools

import pytest
import torch
from torch import nn
from auto_LiRPA import BoundedModule, PerturbationLpNorm

from CTRAIN.bound.ibp import bound_ibp
from CTRAIN.util import construct_c


@pytest.mark.parametrize("labels", [[0, 2], [1, 1]])
def test_margin_specification(labels):
    data = torch.zeros(2, 2, dtype=torch.float64)
    target = torch.tensor(labels)
    logits = torch.tensor([[2., -1., 4.], [3., 5., -2.]], dtype=data.dtype)
    c = construct_c(data, target, 3)
    margins = torch.bmm(c, logits.unsqueeze(-1)).squeeze(-1)
    expected = torch.stack([
        torch.stack([row[label] - row[other] for other in range(3) if other != label])
        for row, label in zip(logits, labels)
    ])
    assert c.dtype == data.dtype and c.device == data.device
    torch.testing.assert_close(margins, expected)


@pytest.mark.parametrize("target", [None, torch.tensor([0, 2])], ids=["logits", "margins"])
def test_affine_ibp_exact_extrema_and_gradients(target):
    model = nn.Sequential(nn.Linear(2, 3)).double().eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1., -2.], [-3., 4.], [2., 1.]]))
        model[0].bias.copy_(torch.tensor([.5, -.5, 1.]))
    low = torch.tensor([[-1., 0.], [0., 2.]], dtype=torch.float64)
    high = torch.tensor([[2., 3.], [1., 2.]], dtype=torch.float64)
    center, radius = (low + high) / 2, (high - low) / 2
    bounded = BoundedModule(model, center, device="cpu")
    lower, upper = bound_ibp(bounded, PerturbationLpNorm(x_L=low, x_U=high),
                             center, target, n_classes=3, bound_upper=True)
    weight, bias = model[0].weight, model[0].bias
    if target is None:
        midpoint = model(center)
        deviation = radius @ weight.abs().T
    else:
        # Compute each true-versus-other affine function independently of C.
        differences = torch.stack([torch.stack([weight[label] - weight[j]
            for j in range(3) if j != label]) for label in target])
        offsets = torch.stack([torch.stack([bias[label] - bias[j]
            for j in range(3) if j != label]) for label in target])
        midpoint = (differences * center[:, None]).sum(-1) + offsets
        deviation = (differences.abs() * radius[:, None]).sum(-1)
    torch.testing.assert_close(lower, midpoint - deviation)
    torch.testing.assert_close(upper, midpoint + deviation)
    lower.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bounded.parameters())
    assert any(p.grad.abs().sum() > 0 for p in bounded.parameters())


def test_relu_ibp_contains_grid_and_zero_radius_is_exact():
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 3)).double().eval()
    center = torch.tensor([[.2, -.1]], dtype=torch.float64)
    bounded = BoundedModule(model, center, device="cpu")
    for radius in (0., .5):
        low, high = center - radius, center + radius
        lower, upper = bound_ibp(bounded, PerturbationLpNorm(x_L=low, x_U=high),
                                 center, None, n_classes=3, bound_upper=True)
        grid = torch.tensor(list(itertools.product(*[
            torch.linspace(low[0, i], high[0, i], 7, dtype=center.dtype).tolist() for i in range(2)
        ])), dtype=center.dtype)
        outputs = model(grid)
        assert (outputs >= lower - 1e-10).all()
        assert (outputs <= upper + 1e-10).all()
        if radius == 0:
            torch.testing.assert_close(lower, model(center))
            torch.testing.assert_close(upper, model(center))
