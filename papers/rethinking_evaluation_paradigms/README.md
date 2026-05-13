# Rethinking Evaluation Paradigms in IBP-based Certified Training

This folder is the branch-local code and artifact area for the paper preprint
"Rethinking Evaluation Paradigms in IBP-based Certified Training". It is not
part of the CTRAIN Python package and is explicitly excluded from source
distributions.

Run the maintained reproduction commands from the repository root:

```bash
cd /storage/work/kaulen/ctrain_dev/CTRAIN
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Contents:

- `mo_hpo/`: maintained CTRAIN-compatible reproduction scripts and commands.
- `submitit_experiments/`: original SLURM/submitit submission scripts from the
  paper codebase.
- `eval/`: plotting, table, front-analysis, and verification-analysis scripts.
- `results/`: result artifacts copied from the paper codebase.
- `ORIGINAL_README.md`: original paper-code README.
- `requirements-paper.txt`: original paper-code requirements snapshot.

Start with `mo_hpo/README.md` for commands to reproduce the HPO, front
aggregation, complete verification, and downstream analyses using the current
CTRAIN codebase.

The copied `submitit_experiments/` scripts are retained for cluster-scale
reproduction. They now use the current CTRAIN loader APIs and the following
environment overrides:

```bash
export CTRAIN_DATA_ROOT=/path/to/data
export CTRAIN_PAPER_RESULTS_ROOT=/path/to/paper/results
```

If those variables are not set, data is read from `../data` relative to this
paper directory's parent (`papers/data`), and results are read/written under
`papers/rethinking_evaluation_paradigms/results`.

For the plotting/table scripts in `eval/`, run from the `eval/` directory
because those scripts use paths relative to their own folder:

```bash
cd papers/rethinking_evaluation_paradigms/eval
python combine_results.py
python plot.py
python tables.py
```
