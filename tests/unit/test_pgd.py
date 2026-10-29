import pytest
import torch
from torch import nn

from CTRAIN.attacks.pgd import pgd_attack


@pytest.fixture
def problem():
    model = nn.Linear(2, 2).double().eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., 0.], [-1., 0.]]))
        model.bias.zero_()
    data = torch.tensor([[.5, .3], [.5, .7]], dtype=torch.float64)
    low = torch.tensor([[-1., .3], [-.5, .6]], dtype=data.dtype)
    high = torch.tensor([[1., .3], [.8, .9]], dtype=data.dtype)
    return dict(model=model, data=data, target=torch.zeros(2, dtype=torch.long),
                x_L=low, x_U=high, device="cpu")


@pytest.mark.parametrize("early_stopping", [False, True])
def test_attack_finds_counterexamples_without_parameter_gradients(problem, early_stopping):
    model = problem["model"]
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 7.)
    model.bias.requires_grad_(False)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    original_data = problem["data"].clone()
    # Attacks must work inside an evaluation caller's no_grad block.
    with torch.no_grad():
        adversarial = pgd_attack(**problem, n_steps=10, step_size=1.,
                                 early_stopping=early_stopping)
    assert (model(adversarial).argmax(1) != problem["target"]).all()
    assert (adversarial >= problem["x_L"]).all() and (adversarial <= problem["x_U"]).all()
    assert adversarial.dtype == original_data.dtype and not adversarial.requires_grad
    torch.testing.assert_close(problem["data"], original_data)
    assert not model.training and not model.bias.requires_grad
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 7.))


def test_zero_width_box_and_zero_steps(problem):
    problem["x_L"] = problem["x_U"] = problem["data"].clone()
    result = pgd_attack(**problem, n_steps=0)
    torch.testing.assert_close(result, problem["data"])


@pytest.mark.parametrize("option,value", [
    ("restarts", 0), ("restarts", True), ("n_steps", -1), ("n_steps", 1.5),
    ("step_size", float("nan")), ("step_size", float("inf")), ("step_size", -.1),
])
def test_invalid_attack_options(problem, option, value):
    with pytest.raises(ValueError):
        pgd_attack(**problem, **{option: value})


@pytest.mark.parametrize("invalid", ["reversed", "nan"])
def test_invalid_attack_boxes(problem, invalid):
    problem["x_L"] = problem["x_U"] + 1 if invalid == "reversed" else torch.full_like(problem["data"], float("nan"))
    with pytest.raises(ValueError, match="finite, ordered"):
        pgd_attack(**problem)
