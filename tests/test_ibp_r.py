"""Runnable CPU regression for IBP-R geometry, gradients and training plumbing."""
from unittest.mock import patch
import copy
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from auto_LiRPA import BoundedModule, PerturbationLpNorm

from CTRAIN import IBPRModelWrapper
from CTRAIN.train.certified.losses.ibp_r import get_ibp_r_loss
from CTRAIN.train.certified.regularisers import get_shi_regulariser
from CTRAIN.train.certified.initialisation import ibp_init_shi
from CTRAIN.model_wrappers.configs import get_config_space, get_combined_config_space


def check():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(1, 1, bias=False), nn.ReLU(),
                          nn.Linear(1, 2, bias=False)).double()
    with torch.no_grad():
        model[0].weight.fill_(2)
        model[2].weight.copy_(torch.tensor([[1.], [-1.]]))
    data = torch.tensor([[.5], [.5]], dtype=torch.double)
    target = torch.tensor([0, 1])
    bounded = BoundedModule(model, data, device="cpu")
    region = PerturbationLpNorm(x_L=torch.tensor([[-1.], [-2.]], dtype=data.dtype),
                                x_U=torch.tensor([[2.], [1.]], dtype=data.dtype))
    # Both preactivation triangles have area 4 at w=2. Masking keeps sample 0.
    for mask, expected in ((False, 4.), (True, 2.)):
        with patch("CTRAIN.train.certified.losses.ibp_r.pgd_attack", return_value=data):
            loss, logits = get_ibp_r_loss(bounded, model, region, data, target,
                                         rho=1., regularisation_mask=mask)
        area = loss - nn.functional.cross_entropy(logits, target)
        torch.testing.assert_close(area, data.new_tensor(expected))
        gradient = torch.autograd.grad(area, model[0].weight)[0]
        torch.testing.assert_close(gradient, data.new_tensor([[expected]]))
    # Fresh bounds must replace previous geometry; stable ReLUs have zero area.
    stable = PerturbationLpNorm(x_L=torch.ones_like(data), x_U=2 * torch.ones_like(data))
    with patch("CTRAIN.train.certified.losses.ibp_r.pgd_attack", return_value=data):
        loss, logits = get_ibp_r_loss(bounded, model, stable, data, target, rho=1.)
    torch.testing.assert_close(loss, nn.functional.cross_entropy(logits, target))
    # Check linear mixing against independently evaluated clean/adversarial CE.
    adversarial = 2 * data
    with patch("CTRAIN.train.certified.losses.ibp_r.pgd_attack", return_value=adversarial):
        mixed, _ = get_ibp_r_loss(bounded, model, region, data, target, kappa=.3, rho=1.)
    expected = .7 * nn.functional.cross_entropy(model(data), target) + .3 * (
        nn.functional.cross_entropy(model(adversarial), target) + 4.)
    torch.testing.assert_close(mixed, expected)
    # Pure clean phase does not attack or bound.
    with patch("CTRAIN.train.certified.losses.ibp_r.pgd_attack", side_effect=AssertionError):
        loss, logits = get_ibp_r_loss(bounded, model, region, data, target, kappa=0)
    torch.testing.assert_close(loss, nn.functional.cross_entropy(logits, target))

    model.train()
    bounded.train()
    model[1].eval()  # Preserve heterogeneous caller modes too.
    modes = {m: m.training for root in (model, bounded) for m in root.modules()}
    for param in bounded.parameters():
        param.grad = torch.ones_like(param)
    grads = {p: p.grad.clone() for p in bounded.parameters()}
    flags = {p: p.requires_grad for p in bounded.parameters()}

    def failing_attack(*args, **kwargs):
        assert not bounded.training and not model.training
        next(bounded.parameters()).requires_grad_(False)
        raise RuntimeError("attack failed")

    with patch("CTRAIN.train.certified.losses.ibp_r.pgd_attack", side_effect=failing_attack):
        try:
            get_ibp_r_loss(bounded, model, region, data, target)
        except RuntimeError as exc:
            assert str(exc) == "attack failed"
        else:
            raise AssertionError("expected attack failure")
    assert all(m.training == mode for m, mode in modes.items())
    assert all(p.requires_grad == flag for p, flag in flags.items())
    for param, grad in grads.items():
        torch.testing.assert_close(param.grad, grad)
    bounded.eval()
    model.eval()
    get_ibp_r_loss(bounded, model, region, data, target, pgd_steps=2)
    assert not bounded.training and not model.training
    for param, grad in grads.items():
        torch.testing.assert_close(param.grad, grad)

    images = torch.tensor([0., .3, .7, 1.]).reshape(4, 1, 1, 1)
    loader = DataLoader(TensorDataset(images, torch.tensor([0, 0, 1, 1])), batch_size=2)
    loader.mean, loader.std = torch.tensor([0.]), torch.tensor([.5])
    loader.min, loader.max, loader.normalised = torch.tensor(0.), torch.tensor(1.), True
    network = nn.Sequential(nn.Flatten(), nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 2))
    wrapper = IBPRModelWrapper(network, (1, 1, 1), eps=.1, num_epochs=3,
                              ramp_up_epochs=1, train_eps_factor=2., rho=.03,
                              regularisation_mask=True, pgd_steps=2, pgd_eps_factor=1.5,
                              relu_upper_retention=.8, population_bn=True, device="cpu",
                              pgd_early_stopping=True, pgd_alpha_decay_factor=.2,
                              pgd_decay_milestones=(1,))
    seen = []
    actual = get_ibp_r_loss

    def record(*args, **kwargs):
        seen.append((kwargs["kappa"], kwargs["eps"].clone()))
        assert kwargs["rho"] == .03 and kwargs["regularisation_mask"]
        assert kwargs["pgd_eps_factor"] == 1.5 and kwargs["relu_upper_retention"] == .8
        assert kwargs["pgd_early_stopping"] and kwargs["pgd_decay_factor"] == .2
        assert kwargs["pgd_decay_checkpoints"] == (1,)
        return actual(*args, **kwargs)

    before = [p.detach().clone() for p in wrapper.bounded_model.parameters()]
    with patch("CTRAIN.train.certified.ibp_r.get_ibp_r_loss", side_effect=record), \
            patch("CTRAIN.train.certified.ibp_r.ibp_init_shi", wraps=ibp_init_shi) as init, \
            patch("CTRAIN.train.certified.ibp_r.get_shi_regulariser", wraps=get_shi_regulariser) as reg:
        assert wrapper.train_model(loader) is wrapper.bounded_model
    assert init.call_count == 1 and reg.call_count == 2
    assert all(call.kwargs["relu_upper_retention"] == .8 and call.kwargs["regularisation_decay"]
               for call in reg.call_args_list)
    assert [k for k, _ in seen] == [0., .5, 1., 1., 1., 1.]
    torch.testing.assert_close(seen[1][1], torch.tensor([[[.2]]]))
    assert abs(wrapper.optimizer.param_groups[0]["lr"] - .01 * .95**3) < 1e-12
    assert any(not torch.equal(p, old) for p, old in zip(wrapper.bounded_model.parameters(), before))
    assert all(torch.isfinite(p).all() for p in wrapper.bounded_model.parameters())
    # Absolute mixing schedule also applies to a partial/resumed invocation.
    seen.clear()
    with patch("CTRAIN.train.certified.ibp_r.get_ibp_r_loss", side_effect=record), \
            patch("CTRAIN.train.certified.ibp_r.ibp_init_shi") as init, \
            patch("CTRAIN.train.certified.ibp_r.get_shi_regulariser") as reg:
        wrapper.train_model(loader, start_epoch=1, end_epoch=2)
    assert [k for k, _ in seen] == [1., 1.]
    assert not init.called and not reg.called
    bn_network = nn.Sequential(nn.BatchNorm2d(1), nn.Flatten(), nn.Linear(1, 3),
                               nn.ReLU(), nn.Linear(3, 2))
    bn_wrapper = IBPRModelWrapper(bn_network, (1, 1, 1), eps=.1, num_epochs=1,
                                 ramp_up_epochs=0, pgd_steps=2,
                                 population_bn=True, device="cpu")
    bn_wrapper.train_model(loader)
    originals = dict(bn_wrapper.original_model.named_buffers())
    for name, buffer in bn_wrapper.bounded_model.named_buffers():
        torch.testing.assert_close(buffer, originals[name])
    assert bn_network[0].momentum == .1
    # Fixed validation epsilon, unshrunk attack region, and best inference weights.
    validation = DataLoader(TensorDataset(images[:2], torch.zeros(2, dtype=torch.long)), batch_size=2)
    for name in ("normalised", "min", "max", "mean", "std"):
        setattr(validation, name, getattr(loader, name))
    selected = []

    def validation_attack(model, data, target, lower, upper, **kwargs):
        assert not model.training
        # Nominal eps .1 / std .5; training uses 2.1 times this radius.
        torch.testing.assert_close(lower, (data - .2).clamp(0, 1))
        torch.testing.assert_close(upper, (data + .2).clamp(0, 1))
        with torch.no_grad():
            predicted = model(data).argmax(1)
        target.copy_(predicted if not selected else 1 - predicted)
        selected.append(copy.deepcopy(model.state_dict()))
        return data

    selection_wrapper = IBPRModelWrapper(copy.deepcopy(network), (1, 1, 1), .1,
                                        num_epochs=2, ramp_up_epochs=0, pgd_steps=1,
                                        device="cpu", relu_upper_retention=.8)
    with patch("CTRAIN.train.certified.ibp_r.pgd_attack", side_effect=validation_attack):
        selection_wrapper.train_model(loader, validation)
    assert len(selected) == 2
    for name, value in selection_wrapper.bounded_model.state_dict().items():
        torch.testing.assert_close(value, selected[0][name])
    assert selection_wrapper.bounded_model.training
    # Config construction and actual HPO wrapper recreation retain fixed options.
    config = get_config_space(wrapper, 100, .1, {"train_eps_factor": 2.1}).get_default_configuration()
    assert "ibp_r:rho" in config
    assert "ibp_r:rho" in get_combined_config_space(100, .1, included_methods=["ibp_r"])
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "nets").mkdir()
        with patch.object(IBPRModelWrapper, "train_model") as train, \
                patch.object(IBPRModelWrapper, "_evaluate_hpo_model", return_value=(.8, .5, .6)), \
                patch("CTRAIN.model_wrappers.ibp_r_model_wrapper.IBPRModelWrapper", wraps=IBPRModelWrapper) as recreated:
            _, metrics = wrapper._hpo_runner(config, 0, 100, loader, validation, directory)
        assert train.call_count == 1 and metrics["cert_acc"] == .5
        options = recreated.call_args.kwargs
        assert options["population_bn"] and options["relu_upper_retention"] == .8
        assert options["shi_init"] and options["shi_reg_weight"] == config["shi_reg_weight"]
        assert options["pgd_early_stopping"] and options["pgd_alpha_decay_factor"] == .2
        assert options["pgd_decay_milestones"] == (1,)
    paper_wrapper = IBPRModelWrapper(copy.deepcopy(network), (1, 1, 1), .1,
                                    num_epochs=2, ramp_up_epochs=1, pgd_steps=1,
                                    shi_init=False, shi_reg_weight=0,
                                    lr_decay_after_ramp_up=True, device="cpu")
    with patch("CTRAIN.train.certified.ibp_r.ibp_init_shi") as init, \
            patch("CTRAIN.train.certified.ibp_r.get_shi_regulariser") as reg:
        paper_wrapper.train_model(loader, end_epoch=1)
        assert paper_wrapper.optimizer.param_groups[0]["lr"] == .01
        paper_wrapper.train_model(loader, start_epoch=1)
    assert not init.called and not reg.called
    assert paper_wrapper.optimizer.param_groups[0]["lr"] == .01 * .95
    for invalid in ({"rho": -1}, {"pgd_steps": 0}, {"train_eps_factor": .5},
                    {"relu_upper_retention": 0}, {"pgd_eps_factor": float("nan")},
                    {"shi_reg_weight": -1}):
        try:
            IBPRModelWrapper(None, (1, 1, 1), .1, device="cpu", **invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {invalid}")
    print("IBP-R checks passed")


if __name__ == "__main__":
    check()
