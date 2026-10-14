# CTRAIN case studies

These studies use the library's models and `PropertyModelWrapper.train_model(...)`.
Training algorithms and bound computations live in CTRAIN. Each folder contains
its experiment entry points, configuration and protocol; matching notebooks
live in `docs/examples/case_studies/`.

## Setup on a new machine

Run from the root of a checkout containing this feature, rather than an older PyPI release.
The commands use Python 3.11 and the PyTorch/NumPy versions used in the CPU checks:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "torch==2.3.1" "torchvision==0.18.1" "numpy==1.26.4"
python -m pip install -e ".[dev]"
git submodule update --init --recursive
ctrain-install-git-deps
```

For notebook execution, also install the Python kernel with
`python -m pip install ipykernel` and select this virtual environment in your notebook editor.

The Git installer supplies the pinned auto_LiRPA, ONNX converter and AutoAttack
revisions. See the repository installation guide for platform requirements.
CPU works for the small execution checks; full MNIST training is substantially
more practical on a supported GPU. Use `--device cuda` explicitly when available.
Run all commands below from the repository root. Existing experiment directories
are rejected: choose a fresh `--output` for each run. Dataset downloads are explicit.

| Study | Folder | Notebook |
|---|---|---|
| Controller safety | [controller](controller/README.md) | [Controller](../docs/examples/case_studies/controller.ipynb) |
| ACAS Xu repair | [acasxu](acasxu/README.md) | [ACAS Xu](../docs/examples/case_studies/acasxu.ipynb) |
| MNIST addition consistency and prediction stability | [mnist_addition](mnist_addition/README.md) | [MNIST addition](../docs/examples/case_studies/mnist_addition.ipynb) |

These folders contain portable Python entry points, without scheduler integration,
trained weights, downloaded data or historical run outputs. The notebooks start
with CPU examples that do not download data. Full experiments are explicit steps.

## Checks

```bash
python -m case_studies.controller.run --smoke --output runs/case_studies/controller_smoke
python -m case_studies.acasxu.run --smoke --output runs/case_studies/acasxu_smoke
python -m case_studies.mnist_addition.run --smoke --output runs/case_studies/addition_smoke
python -m case_studies.mnist_addition.local_robustness --self-test
```

Smoke runs check execution, not benchmark performance. Property certificates
apply to the declared regions and formulas, use floating-point bounds, and are
separate from empirical task accuracy, advisory fidelity or rollout success.
