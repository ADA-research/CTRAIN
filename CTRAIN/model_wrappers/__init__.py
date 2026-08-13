"""Model-wrapper exports, loaded lazily to keep lightweight utilities importable."""

from importlib import import_module


_EXPORTS = {
    "ShiIBPModelWrapper": ".shi_ibp_model_wrapper",
    "CrownIBPModelWrapper": ".crown_ibp_model_wrapper",
    "SABRModelWrapper": ".sabr_model_wrapper",
    "TAPSModelWrapper": ".taps_model_wrapper",
    "STAPSModelWrapper": ".staps_model_wrapper",
    "MTLIBPModelWrapper": ".mtl_ibp_model_wrapper",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
