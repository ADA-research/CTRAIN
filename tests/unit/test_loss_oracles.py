"""Compare every deterministic classifier objective and gradient to affine algebra.

Attack outputs are fixed when checking a loss formula; real searches, geometry,
and optimizer updates are exercised separately by test_methods.py. GLS has a
stochastic optimizer rather than another loss; its estimator has its own oracle.
"""
from unittest.mock import patch

import pytest
import torch
from torch import nn
from auto_LiRPA import BoundedModule, CrossEntropyWrapper, PerturbationLpNorm

from CTRAIN.train.adversarial.losses.pgd import get_pgd_loss
from CTRAIN.train.certified.losses import get_ibp_loss, get_crown_ibp_loss, get_sabr_loss, get_taps_loss
from CTRAIN.train.certified.losses.convex_combinations import get_mtl_ibp_loss, get_cc_ibp_loss, get_exp_ibp_loss
from CTRAIN.train.certified.losses.cc_dist import get_cc_dist_loss
from CTRAIN.train.certified.losses.ibp_r import get_ibp_r_loss
from CTRAIN.train.certified.losses.set_based import get_set_based_loss
from CTRAIN.train.certified.gls import GLSOptimizer


OBJECTIVES = ["PGD", "Shi-IBP", "CROWN-IBP-0", "CROWN-IBP-.4", "CROWN-IBP-1",
              "SABR", "MTL-IBP", "CC-IBP", "Exp-IBP", "CC-Dist", "TAPS", "STAPS",
              "IBP-R", "SetBased"]


def margin_ce(margins):
    return torch.logsumexp(torch.cat([torch.zeros_like(margins[:, :1]), -margins], 1), 1).mean()


@pytest.fixture
def affine():
    model = nn.Sequential(nn.Flatten(), nn.Linear(2, 3)).double().eval()
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[1., -2.], [-3., 4.], [2., 1.]]))
        model[1].bias.copy_(torch.tensor([.5, -.5, 1.]))
    return model


@pytest.mark.parametrize("method", OBJECTIVES)
@pytest.mark.parametrize("eps", [0., .2], ids=["zero-radius", "nonzero-radius"])
def test_objective_value_and_parameter_gradients(method, eps, affine):
    # Asymmetric intervals, including one fixed coordinate.
    low = torch.tensor([[.1, .7], [.5, .1]], dtype=torch.float64).reshape(2, 2, 1, 1)
    high = torch.tensor([[.5, .7], [.6, .5]], dtype=low.dtype).reshape_as(low)
    data = (low + high) / 2
    if eps == 0:
        low, high = data, data
    adv = low + .3 * (high - low)
    target = torch.tensor([0, 2])
    bounded = BoundedModule(affine, data, device="cpu", bound_opts={"param": {"auto_requires_grad": False}})
    bounded.eval()
    for parameter in bounded.parameters():
        parameter.requires_grad_(True)
    region = PerturbationLpNorm(x_L=low, x_U=high)
    criterion = nn.CrossEntropyLoss(reduction="none")
    weight, bias = affine[1].weight, affine[1].bias
    differences = torch.stack([torch.stack([weight[label] - weight[j]
        for j in range(3) if j != label]) for label in target])
    offsets = torch.stack([torch.stack([bias[label] - bias[j]
        for j in range(3) if j != label]) for label in target])

    def exact_margins(lower, upper):
        center, radius = ((lower + upper) / 2).flatten(1), ((upper - lower) / 2).flatten(1)
        return (differences * center[:, None]).sum(-1) + offsets - (differences.abs() * radius[:, None]).sum(-1)

    robust = margin_ce(exact_margins(low, high))
    adversarial = nn.functional.cross_entropy(affine(adv), target)
    alpha = .37
    options = dict(hardened_model=bounded, ptb=region, data=data, target=target,
                   n_classes=3, criterion=criterion)
    if method == "PGD":
        with patch("CTRAIN.train.adversarial.losses.pgd.pgd_attack", return_value=adv):
            actual = get_pgd_loss(**options)
        expected = adversarial
    elif method == "Shi-IBP":
        actual, = get_ibp_loss(**options)
        expected = robust
    elif method.startswith("CROWN-IBP"):
        actual, = get_crown_ibp_loss(**options, beta=float(method.rsplit("-", 1)[1]), return_stats=False)
        expected = robust
    elif method in ("MTL-IBP", "CC-IBP", "Exp-IBP", "CC-Dist"):
        functions = {"MTL-IBP": get_mtl_ibp_loss, "CC-IBP": get_cc_ibp_loss,
                     "Exp-IBP": get_exp_ibp_loss, "CC-Dist": get_cc_dist_loss}
        extra = dict(teacher_features=nn.Flatten(), distillation_weight=.3) if method == "CC-Dist" else {}
        with patch("CTRAIN.train.certified.losses.convex_combinations.pgd_attack", return_value=adv):
            actual, = functions[method](**options, original_model=affine, alpha=alpha, device="cpu", **extra)
        if method == "MTL-IBP":
            expected = alpha * robust + (1 - alpha) * adversarial
        elif method == "Exp-IBP":
            expected = robust ** alpha * adversarial ** (1 - alpha)
        else:
            expected = margin_ce(alpha * exact_margins(low, high) + (1 - alpha) * exact_margins(adv, adv))
            if method == "CC-Dist":
                left = alpha * low + (1 - alpha) * adv
                right = alpha * high + (1 - alpha) * adv
                expected = expected + .3 * torch.maximum((left - data).square(), (right - data).square()).flatten(1).sum(1).mean()
    elif method == "IBP-R":
        with patch("CTRAIN.train.certified.losses.ibp_r.pgd_attack", return_value=adv):
            actual, _ = get_ibp_r_loss(bounded, affine, region, data, target, kappa=alpha, rho=.1)
        # An affine model has no ReLU relaxation area to penalize.
        expected = (1 - alpha) * nn.functional.cross_entropy(affine(data), target) + alpha * adversarial
    elif method == "SABR":
        ratio = .6
        with patch("CTRAIN.bound.sabr.pgd_attack", return_value=adv):
            actual, = get_sabr_loss(bounded, affine, data, target, eps, ratio, criterion,
                device="cpu", n_classes=3, x_L=low, x_U=high, data_min=0., data_max=1.)
        tau = ratio * (high - low) / 2
        center = adv.clamp(low + tau, high - tau)
        expected = margin_ce(exact_margins(center - tau, center + tau))
    elif method in ("TAPS", "STAPS"):
        features = BoundedModule(affine[:1], data, device="cpu")
        classifier = BoundedModule(affine[1:], data.flatten(1), device="cpu")
        sabr = dict(hardened_model=bounded, original_model=affine, data=data, target=target,
                    eps=eps, subselection_ratio=1., device="cpu", n_classes=3,
                    x_L=low, x_U=high, data_min=0., data_max=1., n_steps=2)
        with patch("CTRAIN.bound.sabr.pgd_attack", return_value=adv):
            actual, = get_taps_loss(affine, bounded, [features, classifier], criterion,
                data, target, 3, region, device="cpu", pgd_steps=10, pgd_step_size=1.,
                gradient_expansion_alpha=1., propagation="SABR" if method == "STAPS" else "IBP", sabr_args=sabr)
        # With an identity feature block and affine classifier, latent search
        # reaches exact margin extrema. The TAPS objective is their loss product.
        expected = robust.square()
    else:
        tau = .25
        actual = get_set_based_loss(affine, data, target, eps, (high - low) / 2,
                                    low, high, tau=tau)
        if eps == 0:
            expected = nn.functional.cross_entropy(affine(data), target)
        else:
            generators = ((high - low) / 2).flatten(1)[:, None, :] * weight[None, :, :]
            norm = torch.linalg.vector_norm(generators.flatten(1), dim=1).mean()
            expected = (1 - tau) * nn.functional.cross_entropy(affine(data), target) + tau * norm / (3 * eps)
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(actual, (weight, bias), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, (weight, bias))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("beta", [0., .4, 1.])
def test_fused_crown_zero_radius_value_and_gradients(beta, affine):
    # Installed LiRPA's fused Exp backward constructs float32 coefficients.
    affine = affine.float()
    data = torch.tensor([[.2, .7], [.8, .3]]).reshape(2, 2, 1, 1)
    labels = torch.tensor([0, 2])
    fused = BoundedModule(CrossEntropyWrapper(affine), (data, labels), device="cpu",
                          bound_opts={"param": {"auto_requires_grad": False}})
    for parameter in fused.parameters():
        parameter.requires_grad_(True)
    fused.eval()
    fused(data, labels)
    actual, = get_crown_ibp_loss(fused, PerturbationLpNorm(x_L=data, x_U=data), data, labels,
        3, nn.CrossEntropyLoss(), beta, loss_fusion=True, return_stats=False)
    expected = nn.functional.cross_entropy(affine(data), labels)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual, affine.parameters(), retain_graph=True),
                    torch.autograd.grad(expected, affine.parameters())):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("estimator,expected", [("rgs", 1.5), ("pgpe", 1.8)])
def test_gls_reference_estimator_update_and_failure_restoration(estimator, expected):
    model = nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.fill_(2.)
    optimizer = GLSOptimizer(model, torch.optim.SGD, lr=.1, estimator=estimator,
                             samples=2, sigma=.5, sigma_lr=0.)
    # Fixed unit Gaussian draws give a hand-computable update for each estimator.
    with patch("torch.randn_like", side_effect=torch.ones_like):
        optimizer.step(lambda: model.weight.square().sum())
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, expected))
    before = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, 3.)
    def fail():
        raise RuntimeError("injected")
    with pytest.raises(RuntimeError, match="injected"):
        optimizer.step(fail)
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 3.), rtol=0, atol=0)
