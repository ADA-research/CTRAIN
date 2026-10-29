"""Public imports must retain class identity across compatibility paths."""
from importlib import import_module

import pytest
from torch import nn

import CTRAIN
from CTRAIN.util.util import preserve_model_state


@pytest.mark.parametrize("name,module", [
    ("PGDModelWrapper", "pgd_model_wrapper"),
    ("ShiIBPModelWrapper", "shi_ibp_model_wrapper"),
    ("CrownIBPModelWrapper", "crown_ibp_model_wrapper"),
    ("SABRModelWrapper", "sabr_model_wrapper"),
    ("MTLIBPModelWrapper", "mtl_ibp_model_wrapper"),
    ("TAPSModelWrapper", "taps_model_wrapper"),
    ("STAPSModelWrapper", "staps_model_wrapper"),
    ("CCIBPModelWrapper", "cc_ibp_model_wrapper"),
    ("ExpIBPModelWrapper", "exp_ibp_model_wrapper"),
    ("CCDistModelWrapper", "cc_dist_model_wrapper"),
    ("ExpressiveIBPModelWrapper", "expressive_ibp_model_wrapper"),
    ("GLSModelWrapper", "gls_model_wrapper"),
    ("IBPRModelWrapper", "ibp_r_model_wrapper"),
    ("SetBasedModelWrapper", "set_based_model_wrapper"),
    ("PropertyModelWrapper", "property_model_wrapper"),
])
def test_wrapper_import_identity(name, module):
    public = getattr(CTRAIN, name)
    group = "adversarial" if name == "PGDModelWrapper" else "certified"
    for path in ("CTRAIN.model_wrappers", f"CTRAIN.model_wrappers.{group}",
                 f"CTRAIN.model_wrappers.{module}", f"CTRAIN.model_wrappers.{group}.{module}"):
        assert getattr(import_module(path), name) is public
    assert issubclass(public, nn.Module)


@pytest.mark.parametrize("fail", [False, True], ids=["success", "exception"])
def test_restore_mixed_modes_and_gradient_flags(fail):
    model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.Linear(3, 2))
    model.train()
    model[1].eval()
    model[0].weight.requires_grad_(False)
    modes = [module.training for module in model.modules()]
    flags = [parameter.requires_grad for parameter in model.parameters()]
    try:
        # Repeated models exercise shared references in full/split networks.
        with preserve_model_state(model, model[0]):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(not parameter.requires_grad)
            if fail:
                raise RuntimeError("computation failed")
    except RuntimeError as error:
        assert fail and str(error) == "computation failed"
    assert [module.training for module in model.modules()] == modes
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
