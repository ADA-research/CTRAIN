"""Compatibility alias for the canonical certified wrapper module."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module("CTRAIN.model_wrappers.certified.ibp_r_model_wrapper")
