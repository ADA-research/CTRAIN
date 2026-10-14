# Case studies

The case studies demonstrate general-property training and verification through
CTRAIN's public APIs. Each notebook starts with a small CPU example without
automatic data downloads; full experiments are explicit steps.

## Install from this checkout

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

Run CLI commands from the repository root. Open the notebooks with the environment
containing CTRAIN; their setup cell locates the checkout from the current directory.
Choose a fresh output directory for each CLI run. See the source
`case_studies/README.md` and each study's README for configurations and input artifacts.

- [Controller safety](controller.ipynb): bounded actions and robust one-step invariance.
- [ACAS Xu repair](acasxu.ipynb): joint output-property repair and advisory fidelity.
- [MNIST addition](mnist_addition.ipynb): arithmetic consistency, HPO and prediction stability.

The general-property methods are experimental adaptations. Synthetic execution
checks are not benchmark reproductions. Floating-point bounds and empirical task
metrics have distinct meanings; unknown verification outcomes remain unresolved.
