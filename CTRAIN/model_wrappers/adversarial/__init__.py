"""Adversarial training wrappers, loaded lazily."""

from importlib import import_module


_EXPORTS = {
    "PGDModelWrapper": ".pgd_model_wrapper",
}

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
