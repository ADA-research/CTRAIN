"""Shared HPO options and source provenance for runnable property examples."""
import copy
import hashlib
import subprocess
from pathlib import Path


PROPERTY_METHOD_SEARCH_SPACE = {
    'MTL-IBP': {'alpha': [0., 1.]},
    'SABR': {'subselection_ratio': [.01, 1.]},
}


def sample_training_options(trial, method, options):
    options = copy.deepcopy(options)
    for name, limits in PROPERTY_METHOD_SEARCH_SPACE.get(method, {}).items():
        options[name] = trial.suggest_float(name, *limits, log=name == 'subselection_ratio')
    return options


def source_provenance():
    root = Path(__file__).resolve().parents[1]
    paths = [root / name for name in (
        'CTRAIN/properties/__init__.py', 'CTRAIN/bound/properties.py',
        'CTRAIN/data_loaders/acasxu.py', 'CTRAIN/util/onnx.py',
        'CTRAIN/data_loaders/data_loaders.py',
        'CTRAIN/data_loaders/vnnlib.py', 'CTRAIN/properties/vnnlib.py',
        'CTRAIN/eval/properties.py', 'CTRAIN/model_wrappers/certified/property_model_wrapper.py',
        'CTRAIN/train/certified/losses/property_methods.py', 'CTRAIN/properties/training.py', 'CTRAIN/attacks/pgd.py',
        'CTRAIN/bound/sabr.py', 'CTRAIN/train/certified/gls.py',
        'CTRAIN/bound/util.py', 'CTRAIN/train/certified/losses/util.py',
        'CTRAIN/train/certified/losses/ibp_r.py',
        'CTRAIN/train/certified/losses/convex_combinations.py',
        'CTRAIN/train/certified/properties.py', 'CTRAIN/train/certified/losses/properties.py',
        'CTRAIN/model_definitions/models_properties.py',
        'case_studies/common.py',
        'case_studies/controller/run.py', 'case_studies/acasxu/run.py',
        'case_studies/mnist_addition/run.py',
        'case_studies/mnist_addition/data.py',
        'case_studies/mnist_addition/config.json',
        'case_studies/acasxu/evaluate.py', 'case_studies/acasxu/suite.py')]
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(commit=commit, sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in paths})
