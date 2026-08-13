_MODEL_WRAPPER_EXPORTS = {
    "ShiIBPModelWrapper": "CTRAIN.model_wrappers",
    "SABRModelWrapper": "CTRAIN.model_wrappers",
    "CrownIBPModelWrapper": "CTRAIN.model_wrappers",
    "MTLIBPModelWrapper": "CTRAIN.model_wrappers",
    "TAPSModelWrapper": "CTRAIN.model_wrappers",
    "STAPSModelWrapper": "CTRAIN.model_wrappers",
}


def __getattr__(name):
    if name not in _MODEL_WRAPPER_EXPORTS:
        raise AttributeError(f"module 'CTRAIN' has no attribute {name!r}")

    try:
        module = __import__(_MODEL_WRAPPER_EXPORTS[name], fromlist=[name])
    except (ImportError, ModuleNotFoundError) as exc:
        raise ModuleNotFoundError(
            "CTRAIN model wrappers require Git-hosted dependencies that PyPI "
            "cannot install automatically. Run `ctrain-install-git-deps` "
            "before importing a model wrapper."
        ) from exc
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = sorted(_MODEL_WRAPPER_EXPORTS)
