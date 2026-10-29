"""Small CPU check: hybrid options, ReLU soundness and BN export."""
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from auto_LiRPA import BoundedModule, PerturbationLpNorm
from auto_LiRPA.operators.relu import BoundRelu

from CTRAIN.train.certified.util import synchronise_bn
from CTRAIN.bound.ibp import bound_ibp
from CTRAIN.bound.util import get_pgd_ptb, validate_hybrid_options
from CTRAIN.bound.taps import bound_taps, _get_pivotal_points
from CTRAIN.attacks import pgd_attack
from CTRAIN.model_wrappers.shi_ibp_model_wrapper import ShiIBPModelWrapper
from CTRAIN.model_wrappers.crown_ibp_model_wrapper import CrownIBPModelWrapper
from CTRAIN.model_wrappers.sabr_model_wrapper import SABRModelWrapper
from CTRAIN.model_wrappers.mtl_ibp_model_wrapper import MTLIBPModelWrapper
from CTRAIN.model_wrappers.taps_model_wrapper import TAPSModelWrapper
from CTRAIN.model_wrappers.staps_model_wrapper import STAPSModelWrapper


def check():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.BatchNorm2d(3), nn.ReLU(), nn.Flatten(), nn.Linear(3, 2))

        def forward(self, data):
            return self.layers(data)

    original = Toy().eval()
    data = torch.arange(12, dtype=torch.float32).reshape(4, 3, 1, 1)
    full = BoundedModule(original, data[:2], device="cpu")
    features, classifier = nn.Sequential(original.layers[:3]), nn.Sequential(original.layers[3:])
    full.original_blocks = [features, classifier]
    full.bounded_blocks = [BoundedModule(features, data[:2], device="cpu"),
                           BoundedModule(classifier, features(data[:2]), device="cpu")]
    relu_model = BoundedModule(nn.ReLU(), torch.zeros(1, 1), device="cpu")
    relu_data = torch.tensor([[.5]])
    region = PerturbationLpNorm(x_L=torch.tensor([[-1.]]), x_U=torch.tensor([[2.]]))
    _, upper = bound_ibp(relu_model, region, relu_data, None, bound_upper=True, relu_upper_retention=.8)
    assert torch.allclose(upper, torch.tensor([[1.6]]))
    _, upper = bound_ibp(relu_model, region, relu_data, None, bound_upper=True)
    assert upper.item() == 2
    try:
        with patch.object(relu_model, "compute_bounds", side_effect=RuntimeError("bound failed")):
            bound_ibp(relu_model, region, relu_data, None, relu_upper_retention=.8)
    except RuntimeError:
        pass
    assert all("interval_propagate" not in node.__dict__
               for node in relu_model.modules() if isinstance(node, BoundRelu))
    # Attack expansion must recover the unclipped radius at pixel boundaries.
    edge = torch.tensor([[0., .5, 1.]])
    nominal = PerturbationLpNorm(eps=.1, x_L=(edge - .1).clamp(0, 1), x_U=(edge + .1).clamp(0, 1))
    expanded = get_pgd_ptb(nominal, edge, 2, 0., 1.)
    assert torch.allclose(expanded.x_L, torch.tensor([[0., .3, .8]]))
    assert torch.allclose(expanded.x_U, torch.tensor([[.2, .7, 1.]]))
    assert torch.allclose(nominal.x_U, torch.tensor([[.1, .6, 1.]]))
    loader = DataLoader(TensorDataset(data, torch.zeros(4, dtype=torch.long)), batch_size=2)
    synchronise_bn(original, full, loader, "cpu", population=True)
    assert not original.training and original.layers[0].momentum == .1
    expected = data.mean((0, 2, 3))
    assert torch.allclose(original.layers[0].running_mean, expected)
    assert torch.allclose(dict(full.named_buffers())["layers.0.running_mean"], expected)
    assert torch.allclose(dict(full.bounded_blocks[0].named_buffers())["0.0.running_mean"], expected)
    loader.std = torch.ones(3)
    loader.min, loader.max = torch.zeros(3, 1, 1), torch.full((3, 1, 1), 12.)
    loader.mean, loader.normalised = torch.zeros(3), True
    for model in (original, full, *full.bounded_blocks):
        model.eval()
    image_region = PerturbationLpNorm(eps=.1, x_L=(data - .1).clamp(0, 12), x_U=(data + .1).clamp(0, 12))
    ranges = []
    for factor in (1., 2.):
        with patch("CTRAIN.bound.taps._get_pivotal_points", wraps=_get_pivotal_points) as search:
            bound_taps(original, full, full.bounded_blocks, data, torch.zeros(4, dtype=torch.long),
                       2, image_region, device="cpu", pgd_steps=1, pgd_eps_factor=factor)
            ranges.append(search.call_args.args[1:3])
    assert torch.allclose(ranges[1][1] - ranges[1][0], 2 * (ranges[0][1] - ranges[0][0]))
    assert torch.allclose(ranges[1][1] + ranges[1][0], ranges[0][1] + ranges[0][0])
    for retention, factor in ((0., 1.), (1.1, 1.), (1., 0.), (1., float("nan"))):
        try:
            validate_hybrid_options(retention, factor)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid training option accepted")
    # Exercise public wrapper calls, per-epoch recalibration, and evaluation eps.
    for wrapper_type, method in ((TAPSModelWrapper, "taps"), (STAPSModelWrapper, "staps")):
        wrapper = wrapper_type(Toy(), input_shape=(1, 3, 1, 1), eps=.1, train_eps_factor=1.5,
                               num_epochs=10, warm_up_epochs=0, ramp_up_epochs=8,
                               block_sizes=[3, 1], device="cpu", population_bn=True,
                               relu_upper_retention=.8, pgd_steps=1, pgd_eps_factor=1.5,
                               **({"sabr_pgd_steps": 1} if method == "staps" else {}))
        module = f"CTRAIN.train.certified.{method}"
        from CTRAIN.train.certified.util import synchronise_bn as sync
        with patch(f"{module}.synchronise_bn", wraps=sync) as recalibrate:
            wrapper.train_model(loader, end_epoch=1)
            assert recalibrate.call_count == 1
            assert recalibrate.call_args.args[-1] is True
        for bounded in (wrapper.bounded_model, *wrapper.bounded_model.bounded_blocks):
            assert all("interval_propagate" not in node.__dict__
                       for node in bounded.modules() if isinstance(node, BoundRelu))
        # Skip directly to full-radius training so selection must use eps=.1, not .15.
        with patch(f"{module}.evaluate_taps_proxy", return_value=.5) as evaluate:
            wrapper.train_model(loader, val_loader=loader, start_epoch=9, end_epoch=10)
            assert evaluate.call_count == 1
            assert torch.allclose(torch.as_tensor(evaluate.call_args.args[3]), torch.tensor(.1))
        for name, value in wrapper.bounded_model.named_buffers():
            assert torch.equal(value, dict(wrapper.original_model.named_buffers())[name])
    # All other wrappers must recalibrate before saving; include CROWN loss fusion.
    from CTRAIN.train.certified.util import synchronise_bn as sync
    for wrapper_type, method, extra in (
        (ShiIBPModelWrapper, "shi_ibp", {}),
        (CrownIBPModelWrapper, "crown_ibp", {"loss_fusion": False}),
        (CrownIBPModelWrapper, "crown_ibp", {"loss_fusion": True}),
        (SABRModelWrapper, "sabr", {"pgd_steps": 1, "pgd_eps_factor": 2, "relu_upper_retention": .8}),
        (MTLIBPModelWrapper, "mtl_ibp", {"pgd_steps": 1, "pgd_eps_factor": 2, "relu_upper_retention": .8}),
    ):
        wrapper = wrapper_type(Toy(), input_shape=(1, 3, 1, 1), eps=.1,
                               num_epochs=10, warm_up_epochs=0, ramp_up_epochs=8,
                               device="cpu", population_bn=True, **extra)
        attack_path = ("CTRAIN.train.certified.losses.convex_combinations.pgd_attack"
                       if method == "mtl_ibp" else "CTRAIN.bound.sabr.pgd_attack")
        sync_module = "expressive_ibp" if method == "mtl_ibp" else method
        with patch(f"CTRAIN.train.certified.{sync_module}.synchronise_bn", wraps=sync) as recalibrate, \
             patch(attack_path, wraps=pgd_attack) as attack:
            wrapper.train_model(loader, start_epoch=9, end_epoch=10)
            assert recalibrate.call_count == 1 and recalibrate.call_args.args[-1] is True
            if method in ("sabr", "mtl_ibp"):
                args = attack.call_args.kwargs
                assert torch.allclose(args["x_L"], (args["data"] - .2).clamp(0, 12))
                assert torch.allclose(args["x_U"], (args["data"] + .2).clamp(0, 12))
        for name, value in wrapper.bounded_model.named_buffers():
            assert torch.equal(value, dict(wrapper.original_model.named_buffers())[name])
        if getattr(wrapper, "loss_fusion", False):
            for name, value in wrapper.loss_fusion_model.named_buffers():
                assert torch.equal(value, dict(wrapper.original_model.named_buffers())[name.removeprefix("model.")])
    print("Training option and reproduction checks passed")


if __name__ == "__main__":
    check()
