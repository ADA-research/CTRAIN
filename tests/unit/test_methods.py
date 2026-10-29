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


@pytest.mark.parametrize("normalised", [False, True], ids=["raw", "normalised"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_training_gradients_checkpoints_and_sound_evaluation(method, tmp_path, normalised, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    data_loader = loader(normalised)
    wrapper = make(method, tmp_path, device=device)
    before = [parameter.detach().clone() for parameter in wrapper.parameters()]
    updates = []
    def checked_step(optimizer, args, kwargs):
        gradients = [parameter.grad for parameter in wrapper.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().sum() > 0 for gradient in gradients)
        assert all(torch.isfinite(parameter).all() for parameter in wrapper.parameters())
        updates.append(1)

    # The real optimizer and all real losses/bounds run through warm-up and ramp-up.
    with wrapper.optimizer.register_step_post_hook(checked_step):
        trained = wrapper.train_model(data_loader)
    assert trained is getattr(wrapper, "bounded_model", wrapper.original_model)
    assert len(updates) == 3 * len(data_loader)
    assert any(not torch.equal(old, parameter) for old, parameter in zip(before, wrapper.parameters()))
    saved = checkpoints(tmp_path)
    assert [state["epoch"] for state in saved] == [1, 2, 3]
    assert all(torch.isfinite(torch.as_tensor(state["loss"])) for state in saved)
    for state in saved:
        assert "optimizer_state_dict" in state and "lr_scheduler_state_dict" in state
    assert_tree_equal(wrapper.state_dict(), saved[-1]["model_state_dict"])

    wrapper.eval()
    images, labels = data_loader.dataset.tensors
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        output = wrapper(images)
    assert output.shape == (4, 3) and torch.isfinite(output).all()
    restored = make(method, device=device)
    restored.load_state_dict(saved[-1]["model_state_dict"])
    restored.eval()
    torch.testing.assert_close(restored(images), output, rtol=0, atol=0)

    # Compare certification after training to a freshly converted sound graph.
    native = copy.deepcopy(wrapper.original_model).eval()
    reference = BoundedModule(native, images, device=device)
    reference.eval()
    if method[0] == "PGD":
        actual = wrapper._evaluation_wrapper().bounded_model
    elif method[0].startswith("SetBased"):
        actual = wrapper._verification_wrapper().bounded_model
    else:
        actual = wrapper.bounded_model
    actual.eval()
    radius = .1 / data_loader.std.to(images)[:, None, None] if normalised else .1
    low = (images - radius).clamp(data_loader.min.to(images), data_loader.max.to(images))
    high = (images + radius).clamp(data_loader.min.to(images), data_loader.max.to(images))
    region = PerturbationLpNorm(x_L=low, x_U=high)
    modes = [module.training for module in wrapper.modules()]
    buffers = [buffer.clone() for buffer in wrapper.buffers()]
    lower, _ = bound_ibp(actual, region, images, labels, 3)
    expected, _ = bound_ibp(reference, PerturbationLpNorm(x_L=low, x_U=high), images, labels, 3)
    torch.testing.assert_close(lower, expected)
    assert modes == [module.training for module in wrapper.modules()]
    assert all(torch.equal(before, after) for before, after in zip(buffers, wrapper.buffers()))
    margins = torch.stack([torch.stack([row[label] - row[other]
        for other in range(3) if other != label]) for row, label in zip(output, labels)])
    assert (lower <= margins + 1e-5).all()


@pytest.mark.parametrize("population", [False, True], ids=["batch-bn", "population-bn"])
def test_bn_buffers_and_momentum(method, population):
    data_loader = loader()
    wrapper = make(method, population_bn=population)
    wrapper.train_model(data_loader)
    original = dict(wrapper.original_model.named_buffers())
    assert wrapper.original_model.layers[0].momentum == .1
    if population:
        expected = data_loader.dataset.tensors[0].mean((0, 2, 3))
        torch.testing.assert_close(original["layers.0.running_mean"], expected)
    if hasattr(wrapper, "bounded_model"):
        for name, value in wrapper.bounded_model.named_buffers():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        for source, model in zip(getattr(wrapper.bounded_model, "original_blocks", []),
                                 getattr(wrapper.bounded_model, "bounded_blocks", [])):
            for name, value in model.named_buffers():
                torch.testing.assert_close(value, dict(source.named_buffers())[name], rtol=0, atol=0)
    if getattr(wrapper, "loss_fusion", False):
        for name, value in wrapper.loss_fusion_model.named_buffers():
            torch.testing.assert_close(value, original[name.removeprefix("model.")], rtol=0, atol=0)


def test_resume_matches_uninterrupted_trajectory(method, tmp_path):
    data_loader = loader()
    torch.manual_seed(9)
    full = make(method)
    initial = copy.deepcopy(full.state_dict())
    rng = torch.get_rng_state()
    full.train_model(data_loader)

    interrupted = make(method, tmp_path)
    interrupted.load_state_dict(initial)
    # CC-Dist owns an independent frozen teacher; use the same reference features.
    if method[0] == "CC-Dist":
        interrupted.teacher_features.load_state_dict(full.teacher_features.state_dict())
    torch.set_rng_state(rng)
    interrupted.train_model(data_loader, end_epoch=2)
    resume_rng = torch.get_rng_state()
    saved_path = next(path for path in tmp_path.glob("*.pt")
                      if torch.load(path, weights_only=False)["epoch"] == 2)
    resumed = make(method)
    if method[0] == "CC-Dist":
        resumed.teacher_features.load_state_dict(full.teacher_features.state_dict())
    torch.set_rng_state(resume_rng)
    resumed.resume_from_checkpoint(saved_path, data_loader)
    # Legacy trajectories omit RNG state: supply it explicitly, without claiming
    # that their checkpoints reproduce arbitrary shuffled/random data loaders.
    assert_tree_equal(resumed.state_dict(), full.state_dict())
    assert_tree_equal(resumed.optimizer.state_dict(), full.optimizer.state_dict())
    assert_tree_equal(resumed.lr_scheduler.state_dict(), full.lr_scheduler.state_dict())


def test_bn_recalibration_failure_restores_caller_state(method):
    wrapper = make(method)
    wrapper.eval()
    wrapper.original_model.layers[0].train()
    original = wrapper.original_model
    bounded = getattr(wrapper, "bounded_model", original)
    models = [original, bounded, *getattr(bounded, "bounded_blocks", [])]
    if getattr(wrapper, "loss_fusion", False):
        models.append(wrapper.loss_fusion_model)
    modes = [(module, module.training) for model in models for module in model.modules()]
    momenta = [(module, module.momentum) for model in models for module in model.modules()
               if hasattr(module, "momentum")]

    class BrokenLoader:
        def __iter__(self):
            yield next(iter(loader()))
            raise RuntimeError("injected loader failure")

    with pytest.raises(RuntimeError, match="injected loader failure"):
        synchronise_bn(original, bounded, BrokenLoader(), "cpu", population=True,
                       loss_fusion_model=getattr(wrapper, "loss_fusion_model", None))
    assert all(module.training == mode for module, mode in modes)
    assert all(module.momentum == momentum for module, momentum in momenta)


def test_method_matrix_covers_public_classifiers():
    assert {case[1] for case in METHODS} == set(CTRAIN.__all__) - {
        "ExpressiveIBPModelWrapper", "PropertyModelWrapper"}


@pytest.mark.parametrize("fail", [False, True], ids=["success", "exception"])
def test_public_evaluation_preserves_modes_buffers_flags_and_gradients(method, fail):
    wrapper = make(method)
    wrapper.eval()
    wrapper.original_model.layers[0].train()  # Restore mixed caller modes too.
    parameters = list(wrapper.parameters())
    parameters[-1].requires_grad_(False)
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 3.)
    modes = [(module, module.training) for module in wrapper.modules()]
    flags = [parameter.requires_grad for parameter in parameters]
    buffers = [buffer.clone() for buffer in wrapper.buffers()]

    def attack(**kwargs):
        assert not kwargs["model"].training
        if fail:
            raise RuntimeError("injected attack failure")
        return pgd_attack(**dict(kwargs, n_steps=2, restarts=1))

    with patch("CTRAIN.eval.eval.pgd_attack", side_effect=attack) as search:
        if fail:
            with pytest.raises(RuntimeError, match="injected attack failure"):
                wrapper.evaluate(loader(), eval_method="IBP", test_samples=4)
        else:
            clean, certified, adversarial = wrapper.evaluate(loader(), eval_method="IBP", test_samples=4)
            assert 0 <= certified <= adversarial <= clean <= 1
        assert search.called
    assert all(module.training == mode for module, mode in modes)
    assert flags == [parameter.requires_grad for parameter in parameters]
    assert all(torch.equal(old, new) for old, new in zip(buffers, wrapper.buffers()))
    for parameter in parameters:
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 3.), rtol=0, atol=0)


@pytest.mark.parametrize("std", [0., float("nan")], ids=["zero-std", "nan-std"])
def test_invalid_normalisation_fails_before_updates(method, std):
    wrapper = make(method)
    data_loader = loader()
    data_loader.normalised = True
    data_loader.std = torch.full((2,), std)
    with patch.object(wrapper.optimizer, "step", side_effect=AssertionError("invalid loader reached optimizer")):
        with pytest.raises(ValueError):
            wrapper.train_model(data_loader)


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


def test_validation_policy_and_checkpoint_trajectory(method, tmp_path):
    name = method[0]
    wrapper = make(method, tmp_path, **({} if name == "PGD" else {
        "warm_up_epochs": 0, "ramp_up_epochs": 0}))
    training, validation = loader(), loader()
    if name in ("PGD", "CC-Dist"):
        with patch.object(wrapper.optimizer, "step", side_effect=AssertionError("unexpected update")):
            with pytest.raises(ValueError, match="validation"):
                wrapper.train_model(training, validation)
        return
    selected = []

    def score(*args, **kwargs):
        selected.append(copy.deepcopy(wrapper.state_dict()))
        value = 1. if len(selected) == 1 else 0.
        if name in ("TAPS", "STAPS"):
            torch.testing.assert_close(torch.as_tensor(args[3]), torch.tensor(.1))
            return value
        if name.startswith("GLS"):
            torch.testing.assert_close(args[2], torch.tensor([[[.1]]]))
            return value, {}
        if name.startswith("SetBased"):
            torch.testing.assert_close(args[2], torch.tensor(.1))
            assert kwargs['method'] == 'FORWARD'
            assert 'error_propagation' not in kwargs
            return value, {}
        # IBP-R selects by adversarial accuracy, not certified accuracy. Control
        # the validation labels only; the real model still evaluates the points.
        model, data, target, lower, upper = args[:5]
        torch.testing.assert_close(lower, (data - .1).clamp(validation.min, validation.max))
        torch.testing.assert_close(upper, (data + .1).clamp(validation.min, validation.max))
        with torch.no_grad():
            predicted = model(data).argmax(1)
        target.copy_(predicted if len(selected) <= len(validation) else (predicted + 1) % 3)
        return data

    selection_path = (
        f"CTRAIN.train.certified.{name.lower()}.evaluate_taps_proxy" if name in ("TAPS", "STAPS") else
        "CTRAIN.eval.eval_certified" if name.startswith("GLS") else
        "CTRAIN.train.certified.set_based.eval_certified" if name.startswith("SetBased") else
        "CTRAIN.train.certified.ibp_r.pgd_attack" if name == "IBP-R" else None)
    if selection_path is None:
        wrapper.train_model(training, validation)
        # These trainers accept validation loaders but return final weights.
        assert_tree_equal(wrapper.state_dict(), checkpoints(tmp_path)[-1]["model_state_dict"])
    else:
        with patch(selection_path, side_effect=score):
            wrapper.train_model(training, validation)
        assert len(selected) == 3 * (len(validation) if name == "IBP-R" else 1)
        best = selected[len(validation) - 1] if name == "IBP-R" else selected[0]
        assert_tree_equal(wrapper.state_dict(), best)
        saved = checkpoints(tmp_path)[-1]
        assert any(not torch.equal(value, saved["model_state_dict"][key])
                   for key, value in wrapper.state_dict().items())
        # Restoring best inference weights must not rewind optimizer/LR state.
        assert_tree_equal(wrapper.optimizer.state_dict(), saved["optimizer_state_dict"])
        assert_tree_equal(wrapper.lr_scheduler.state_dict(), saved["lr_scheduler_state_dict"])


@pytest.mark.parametrize("case", [case for case in METHODS if case[0] != "PGD"], ids=lambda case: case[0])
def test_hpo_recreation_keeps_shared_and_method_options(case, tmp_path):
    options = dict(population_bn=True)
    if case[0] in HYBRIDS:
        options["relu_upper_retention"] = .8
    wrapper = make(case, **options)
    config = dict(get_config_space(wrapper, 3, .1).get_default_configuration())
    if case[0] in ("TAPS", "STAPS"):
        config["taps:block_split_point"] = .5
    (tmp_path / "nets").mkdir()
    recreated = []

    def train(instance, *args, **kwargs):
        recreated.append(instance)

    with patch.object(type(wrapper), "train_model", autospec=True, side_effect=train), \
         patch.object(type(wrapper), "_evaluate_hpo_model", return_value=(.8, .6, .7)):
        loss, metrics = wrapper._hpo_runner(config, 7, 3, loader(), loader(), tmp_path)
    assert loss == pytest.approx(-2.1) and metrics == dict(nat_acc=.8, cert_acc=.6, adv_acc=.7)
    assert len(recreated) == 1
    candidate = recreated[0]
    assert candidate.population_bn is True
    assert candidate.eps == wrapper.eps and candidate.num_epochs == 3
    assert candidate.train_eps == wrapper.eps * config["train_eps_factor"]
    if case[0] in HYBRIDS:
        assert candidate.relu_upper_retention == .8
    if hasattr(wrapper, "block_sizes"):
        assert candidate.block_sizes == (2, 2)
    if hasattr(wrapper, "loss_fusion"):
        assert candidate.loss_fusion == wrapper.loss_fusion
    if case[0].startswith("SetBased"):
        for key in ("input_set", "error_propagation", "num_generators", "propagation_batch_size", "pgd_eps_factor"):
            assert getattr(candidate, key) == getattr(wrapper, key)
    if case[0].startswith("GLS"):
        assert candidate.optimizer.estimator == wrapper.optimizer.estimator
    if case[0] == "CC-Dist":
        assert candidate.teacher_features is not wrapper.teacher_features
        assert_tree_equal(candidate.teacher_features.state_dict(), wrapper.teacher_features.state_dict())


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


@pytest.mark.parametrize("case", [case for case in METHODS if case[0] in HYBRIDS], ids=lambda case: case[0])
@pytest.mark.parametrize("option,value", [
    ("relu_upper_retention", 0.), ("relu_upper_retention", float("nan")),
    ("pgd_eps_factor", 0.), ("pgd_eps_factor", float("nan")),
])
def test_invalid_hybrid_options_fail_before_graph_conversion(case, option, value):
    with patch("CTRAIN.model_wrappers.certified.model_wrapper.BoundedModule",
               side_effect=AssertionError("invalid option reached graph conversion")):
        with pytest.raises(ValueError):
            make(case, **{option: value})


@pytest.mark.parametrize("case", [case for case in METHODS if case[0] in ("TAPS", "STAPS")], ids=lambda case: case[0])
def test_latent_search_factor_changes_search_without_changing_feature_bounds(case):
    from CTRAIN.bound.taps import _get_pivotal_points
    searches, features = [], []
    ibp_module = __import__("CTRAIN.bound.ibp", fromlist=["compute_ibp_bounds"])
    compute = ibp_module.compute_ibp_bounds
    for factor in (1., 1.75):
        torch.manual_seed(21)
        wrapper = make(case, num_epochs=1, warm_up_epochs=0, ramp_up_epochs=0, pgd_eps_factor=factor)

        def points(*args, **kwargs):
            if len(searches) < (1 if factor == 1. else 2):
                searches.append((args[1].detach().clone(), args[2].detach().clone()))
            return _get_pivotal_points(*args, **kwargs)

        def bounds(model, retention, **kwargs):
            result = compute(model, retention, **kwargs)
            if model is wrapper.bounded_model.bounded_blocks[0] and len(features) < (1 if factor == 1. else 2):
                features.append(tuple(value.detach().clone() for value in result))
            return result

        # Hold the SABR center search fixed to isolate the latent expansion.
        def nominal_center(**kwargs):
            return kwargs["data"].detach().clone()

        with patch("CTRAIN.bound.taps._get_pivotal_points", side_effect=points), \
             patch.object(ibp_module, "compute_ibp_bounds", side_effect=bounds), \
             patch("CTRAIN.bound.sabr.pgd_attack", side_effect=nominal_center):
            wrapper.train_model(loader())
    assert len(searches) == len(features) == 2
    torch.testing.assert_close(searches[1][0] + searches[1][1], searches[0][0] + searches[0][1])
    torch.testing.assert_close(searches[1][1] - searches[1][0], 1.75 * (searches[0][1] - searches[0][0]))
    for first, second in zip(features[0], features[1]):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


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
