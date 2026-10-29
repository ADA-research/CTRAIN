"""Smoke-test introductory notebooks using offline fixtures and reduced budgets."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ["certified_training", "model_evaluation", "hyperparameter_optimisation",
             "general_properties", "onnx_vnnlib", "mnist_arithmetic"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_workflow(name, tmp_path):
    path = ROOT / "docs/examples" / f"{name}.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and cell["outputs"] == []
        tree = ast.parse("".join(cell["source"]), filename=f"{path}:cell-{index}")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("case_studies", "reproductions", "examples"))
            elif isinstance(node, ast.Import):
                assert all(not alias.name.startswith(("case_studies", "reproductions", "examples"))
                           for alias in node.names)
    script = r'''
import json
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from CTRAIN.data_loaders import load_mnist
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import ShiIBPModelWrapper, PropertyModelWrapper

torch.set_num_threads(1)

def offline_mnist(*args, **kwargs):
    train, test = load_mnist(normalise=kwargs.get("normalise", True), val_split=False,
                             workers=0, smoke=True)
    return (train, test, test) if kwargs.get("val_split", True) else (train, test)

def small_cnn(*args, **kwargs):
    return CNN7_Shi(*args, **dict(kwargs, width=1, linear_size=4))

class ShortShi(ShiIBPModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **dict(kwargs, num_epochs=3, warm_up_epochs=1,
                         ramp_up_epochs=1, lr_decay_kwargs={"milestones": [2], "gamma": .2},
                         shi_reg_weight=0., l1_reg_weight=0., device="cpu"))

    def evaluate(self, *args, **kwargs):
        return super().evaluate(*args, **dict(kwargs, test_samples=2))

    def hpo(self, *args, **kwargs):
        return super().hpo(*args, **dict(kwargs, budget_trials=2, budget_time=120, eval_samples=2))

class ShortProperty(PropertyModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **dict(kwargs, num_epochs=3, warm_up_epochs=0,
                                      weight_ramp_epochs=1, property_batch_size=2))

def execute(path):
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(json.loads(path.read_text())["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"{path}:cell-{index}", "exec"), namespace)
    return namespace

path = Path(sys.argv[1])
if path.stem == "onnx_vnnlib":
    from CTRAIN.util import export_onnx
    source = torch.nn.Sequential(torch.nn.Linear(2, 2)).eval()
    export_onnx(source, "network.onnx", 1, (2,))
    Path("property.vnnlib").write_text("""
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 -1))
(assert (<= X_0 1))
(assert (>= X_1 -1))
(assert (<= X_1 1))
(assert (<= Y_0 Y_1))
""")

with patch("CTRAIN.data_loaders.load_mnist", offline_mnist), \
     patch("CTRAIN.model_definitions.CNN7_Shi", small_cnn), \
     patch("CTRAIN.model_wrappers.ShiIBPModelWrapper", ShortShi), \
     patch("CTRAIN.model_wrappers.PropertyModelWrapper", ShortProperty):
    if path.stem == "model_evaluation":
        execute(path.with_name("certified_training.ipynb"))
    namespace = execute(path)

if path.stem == "certified_training":
    assert Path("mnist_ibp.pt").is_file()
elif path.stem == "model_evaluation":
    assert all(0 <= namespace[key] <= 1 for key in ("std_acc", "cert_acc", "adv_acc"))
elif path.stem == "hyperparameter_optimisation":
    assert namespace["pareto_front"]
    assert Path(namespace["selected_trial"]["checkpoint_path"]).is_file()
    assert Path("hpo/mnist_ibp/optuna_study.db").is_file()
else:
    report = namespace["after" if path.stem == "onnx_vnnlib" else "report"]
    assert isinstance(report["certified"], bool) and report["cells"]
assert all(torch.isfinite(p).all() for p in namespace["wrapped_model"].parameters())
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg",
               MPLCONFIGDIR=str(tmp_path / "matplotlib"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONOPTIMIZE", None)
    result = subprocess.run([sys.executable, "-c", script, str(path)], cwd=tmp_path,
                            env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, result.stdout
