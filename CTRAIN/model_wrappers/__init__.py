"""Model-wrapper exports, loaded lazily to keep lightweight utilities importable."""

from importlib import import_module


_EXPORTS = {
    "PropertyModelWrapper": ".certified.property_model_wrapper",
    "PGDModelWrapper": ".adversarial.pgd_model_wrapper",
    "CCDistModelWrapper": ".certified.cc_dist_model_wrapper",
    "ExpressiveIBPModelWrapper": ".certified.expressive_ibp_model_wrapper",
    "GLSModelWrapper": ".certified.gls_model_wrapper",
    "IBPRModelWrapper": ".certified.ibp_r_model_wrapper",
    "SetBasedModelWrapper": ".certified.set_based_model_wrapper",
    "ShiIBPModelWrapper": ".certified.shi_ibp_model_wrapper",
    "CrownIBPModelWrapper": ".certified.crown_ibp_model_wrapper",
    "SABRModelWrapper": ".certified.sabr_model_wrapper",
    "TAPSModelWrapper": ".certified.taps_model_wrapper",
    "STAPSModelWrapper": ".certified.staps_model_wrapper",
    "MTLIBPModelWrapper": ".certified.mtl_ibp_model_wrapper",
    "CCIBPModelWrapper": ".certified.cc_ibp_model_wrapper",
    "ExpIBPModelWrapper": ".certified.exp_ibp_model_wrapper",
}


def __getattr__(name):
    """Load the requested public symbol on demand, avoiding eager imports of optional dependencies."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
