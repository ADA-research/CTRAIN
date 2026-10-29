"""Compatibility alias for the canonical adversarial wrapper module."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module("CTRAIN.model_wrappers.adversarial.pgd_model_wrapper")
