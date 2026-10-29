"""The same lifecycle and numerical-state contracts for every classifier trainer."""
import copy
from inspect import signature
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from auto_LiRPA import BoundedModule, PerturbationLpNorm

import CTRAIN
from CTRAIN.bound.ibp import bound_ibp
from CTRAIN.attacks import pgd_attack
from CTRAIN.model_wrappers.configs import get_config_space
from CTRAIN.train.certified.util import synchronise_bn


# ExpressiveIBPModelWrapper is an abstract shared implementation, not a method.
# PropertyModelWrapper has a separate strategy matrix in test_property_methods.py.
METHODS = [
    ("PGD", "PGDModelWrapper", {}),
    ("Shi-IBP", "ShiIBPModelWrapper", {}),
    ("CROWN-IBP", "CrownIBPModelWrapper", {"loss_fusion": False}),
    ("CROWN-IBP-fused", "CrownIBPModelWrapper", {"loss_fusion": True}),
    ("SABR", "SABRModelWrapper", {}),
    ("MTL-IBP", "MTLIBPModelWrapper", {}),
    ("CC-IBP", "CCIBPModelWrapper", {}),
    ("Exp-IBP", "ExpIBPModelWrapper", {}),
    ("CC-Dist", "CCDistModelWrapper", {}),
    ("TAPS", "TAPSModelWrapper", {}),
    ("STAPS", "STAPSModelWrapper", {}),
    ("IBP-R", "IBPRModelWrapper", {}),
    ("GLS-rgs", "GLSModelWrapper", {"estimator": "rgs"}),
    ("GLS-pgpe", "GLSModelWrapper", {"estimator": "pgpe"}),
    *[(f"SetBased-{region}-{errors}", "SetBasedModelWrapper",
       {"input_set": region, "error_propagation": errors})
      for region in ("box", "attack") for errors in ("interval", "generators")],
]
HYBRIDS = {"SABR", "MTL-IBP", "CC-IBP", "Exp-IBP", "CC-Dist", "TAPS", "STAPS", "IBP-R"}


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(), nn.Linear(2, 3))

    def forward(self, data):
        return self.layers(data)


def loader(normalised=False, batch_size=2):
    # Boundaries, unequal clipped widths, non-unit channel standard deviations.
    images = torch.tensor([[0., .2], [.3, 1.], [.7, .4], [1., .8]]).reshape(4, 2, 1, 1)
    mean, std = torch.tensor([.2, .4]), torch.tensor([.5, 2.])
    minimum, maximum = torch.zeros(2, 1, 1), torch.ones(2, 1, 1)
    if normalised:
        images = (images - mean.reshape(1, 2, 1, 1)) / std.reshape(1, 2, 1, 1)
        minimum = (minimum - mean[:, None, None]) / std[:, None, None]
        maximum = (maximum - mean[:, None, None]) / std[:, None, None]
    result = DataLoader(TensorDataset(images, torch.tensor([0, 1, 2, 0])), batch_size=batch_size)
    result.mean, result.std = mean, std
    result.min, result.max, result.normalised = minimum, maximum, normalised
    return result


def make(case, output=None, **changes):
    name, wrapper_name, variant = case
    options = dict(device="cpu", eps=.1, num_epochs=3, lr=.01,
                   checkpoint_save_path=output, checkpoint_save_interval=1,
                   population_bn=False)
    if name != "PGD":
        options.update(warm_up_epochs=1, ramp_up_epochs=1, train_eps_factor=1.5,
                       lr_scheduler_func=torch.optim.lr_scheduler.MultiStepLR,
                       lr_decay_kwargs={"milestones": [2], "gamma": .5})
    if not name.startswith("SetBased"):
        options.update(l1_reg_weight=0.)
        if name != "PGD":
            options["shi_reg_weight"] = 0.
    if name in HYBRIDS or name == "PGD":
        options.update(pgd_steps=2, pgd_restarts=1, pgd_alpha=.5)
    if name in ("TAPS", "STAPS"):
        options["block_sizes"] = (3, 1)
    if name == "STAPS":
        options["sabr_pgd_steps"] = 2
    if name.startswith("GLS"):
        options.update(samples=2, sigma=.01, sigma_lr=.001)
    if name.startswith("SetBased"):
        options.update(propagation_batch_size=2, num_generators=2)
    if name == "CC-Dist":
        options["teacher"] = Classifier().eval()
    options.update(variant)
    options.update(changes)
    return getattr(CTRAIN, wrapper_name)(Classifier(), (2, 1, 1), **options)


@pytest.fixture(params=METHODS, ids=lambda case: case[0])
def method(request):
    return request.param


def checkpoints(directory):
    result = [torch.load(path, map_location="cpu", weights_only=False)
              for path in Path(directory).glob("*.pt")]
    return sorted(result, key=lambda state: state["epoch"])


def assert_tree_equal(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_tree_equal(a, b)
    elif isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, check_device=False)
    else:
        assert actual == expected


@pytest.mark.parametrize("std", [0., float("nan")], ids=["zero-std", "nan-std"])
def test_invalid_normalisation_fails_before_updates(method, std):
    wrapper = make(method)
    data_loader = loader()
    data_loader.normalised = True
    data_loader.std = torch.full((2,), std)
    with patch.object(wrapper.optimizer, "step", side_effect=AssertionError("invalid loader reached optimizer")):
        with pytest.raises(ValueError):
            wrapper.train_model(data_loader)



@pytest.mark.parametrize("case", [case for case in METHODS if case[0] in {
    "Shi-IBP", "CROWN-IBP", "CROWN-IBP-fused", "TAPS", "STAPS"}], ids=lambda case: case[0])
@pytest.mark.parametrize("normalised", [False, True], ids=["raw", "normalised"])
def test_nominal_training_boxes_use_loader_coordinate_system(case, normalised):
    paths = {"Shi-IBP": "shi_ibp.get_ibp_loss", "CROWN-IBP": "crown_ibp.get_crown_ibp_loss",
             "CROWN-IBP-fused": "crown_ibp.get_crown_ibp_loss",
             "TAPS": "taps.get_taps_loss", "STAPS": "staps.get_taps_loss"}
    path = "CTRAIN.train.certified." + paths[case[0]]
    module, name = path.rsplit(".", 1)
    real_loss = getattr(__import__(module, fromlist=[name]), name)
    data_loader = loader(normalised)
    wrapper = make(case, num_epochs=1, warm_up_epochs=0, ramp_up_epochs=0)
    calls = []

    def loss(*args, **kwargs):
        bound = signature(real_loss).bind(*args, **kwargs).arguments
        data, region = bound["data"], bound["ptb"]
        radius = .15 / data_loader.std[:, None, None] if normalised else .15
        torch.testing.assert_close(region.x_L, (data - radius).clamp(data_loader.min, data_loader.max))
        torch.testing.assert_close(region.x_U, (data + radius).clamp(data_loader.min, data_loader.max))
        calls.append(1)
        return real_loss(*args, **kwargs)

    with patch(path, side_effect=loss):
        wrapper.train_model(data_loader)
    assert len(calls) == len(data_loader)



def test_clean_warmup_loss_matches_independent_cross_entropy(method, tmp_path):
    options = dict(eps=0., num_epochs=2)
    if method[0] != "PGD":
        options.update(warm_up_epochs=1, ramp_up_epochs=0)
    wrapper = make(method, tmp_path, **options)
    data_loader = loader(batch_size=4)
    images, labels = data_loader.dataset.tensors
    reference = copy.deepcopy(wrapper.original_model).train()
    expected = nn.functional.cross_entropy(reference(images), labels)
    expected_gradients = torch.autograd.grad(expected, reference.parameters())
    updates = []

    def check_gradients(optimizer, args, kwargs):
        for parameter, gradient in zip(wrapper.original_model.parameters(), expected_gradients):
            # PGPE's score-function estimator is zero with zero noise; RGS
            # reduces to the ordinary gradient when its sampled noise is zero.
            if method[0] == "GLS-pgpe":
                gradient = torch.zeros_like(gradient)
            torch.testing.assert_close(parameter.grad, gradient)
        updates.append(1)

    # Zero noise isolates GLS's base objective from its sampling distribution.
    with patch("torch.randn_like", side_effect=torch.zeros_like), \
         wrapper.optimizer.register_step_post_hook(check_gradients):
        wrapper.train_model(data_loader, end_epoch=1)
    assert len(updates) == 1
    torch.testing.assert_close(torch.as_tensor(checkpoints(tmp_path)[0]["loss"]), expected.detach())



@pytest.mark.parametrize("case", [case for case in METHODS if case[0] in HYBRIDS], ids=lambda case: case[0])
@pytest.mark.parametrize("normalised", [False, True], ids=["raw", "normalised"])
def test_hybrid_options_reach_real_bounds_and_attacks(case, normalised):
    ibp_module = __import__("CTRAIN.bound.ibp", fromlist=["compute_ibp_bounds"])
    compute = ibp_module.compute_ibp_bounds
    data_loader = loader(normalised)
    wrapper = make(case, warm_up_epochs=0, ramp_up_epochs=0, num_epochs=1,
                   pgd_eps_factor=1.75, relu_upper_retention=.8)
    attack_path = ("CTRAIN.bound.sabr.pgd_attack" if case[0] in ("SABR", "STAPS") else
                   "CTRAIN.train.certified.losses.ibp_r.pgd_attack" if case[0] == "IBP-R" else
                   "CTRAIN.train.certified.losses.convex_combinations.pgd_attack")
    # TAPS searches a latent box; its expansion is checked separately below.
    if case[0] == "TAPS":
        attack_path = "CTRAIN.bound.taps._get_pivotal_points"
    module, name = attack_path.rsplit(".", 1)
    real_attack = getattr(__import__(module, fromlist=[name]), name)
    searches = []

    def record_search(*args, **kwargs):
        bound = signature(real_attack).bind(*args, **kwargs).arguments
        if case[0] != "TAPS":
            data = bound["data"]
            radius = .15 * 1.75
            if normalised:
                radius = radius / data_loader.std[:, None, None]
            torch.testing.assert_close(bound["x_L"], (data - radius).clamp(data_loader.min, data_loader.max))
            torch.testing.assert_close(bound["x_U"], (data + radius).clamp(data_loader.min, data_loader.max))
        searches.append(1)
        return real_attack(*args, **kwargs)

    with patch.object(ibp_module, "compute_ibp_bounds", wraps=compute) as bounds, \
         patch(attack_path, side_effect=record_search):
        wrapper.train_model(data_loader)
    assert searches and any(call.args[1] == .8 for call in bounds.call_args_list)
    # Shrinkage is scoped to training: ordinary subsequent bounds stay sound.
    assert all("interval_propagate" not in node.__dict__
               for node in wrapper.bounded_model.modules() if type(node).__name__ == "BoundRelu")
