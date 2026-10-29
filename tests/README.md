# Library tests

Install CTRAIN's development and Git dependencies, then run from the repository root:

```bash
python -m pytest -q
```

In the development workspace, use `../.artifact-venv/bin/python`.
The default suite runs synthetic CPU tests in `tests/unit/` and standalone
regression scripts in separate processes, each with a five-minute timeout.
Required dependency failures fail the suite. GitHub Actions runs this suite
on pull requests, pushes to `dev` and `main`, and manual dispatches.

Useful selections:

```bash
python -m pytest tests/unit -q
python -m pytest -m regression -q
python -m pytest -k resume -q
python -m pytest -m cuda -q
python -m pytest -m verifier -q
```

CUDA-marked lifecycle tests require a GPU and skip when none is available.
The verifier selection runs real bundled NeuralSAT on known SAT/UNSAT affine
properties. Default tests check verifier integration using synthetic results.
Standalone scripts retain their direct command-line entry points.
New pytest modules belong in `tests/unit/`; top-level `tests/test_*.py` scripts
are discovered automatically by `tests/test_regressions.py`.

The suite checks objective values and gradients against independent numerical
references, bound soundness, attack geometry, normalization, BatchNorm state,
validation selection, checkpoints, HPO option propagation, public imports,
and restoration of caller modes and gradients after failures.
The lifecycle matrix covers every concrete public classifier wrapper.

These synthetic checks do not establish full GPU training behavior, paper
reproduction scores, or complete-verifier correctness on benchmark datasets.
