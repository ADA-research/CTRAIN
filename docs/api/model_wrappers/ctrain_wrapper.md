# Model wrapper packages

Certified and hybrid methods live in `CTRAIN.model_wrappers.certified`.
PGD training lives in `CTRAIN.model_wrappers.adversarial`. Implementations use
the existing classes; legacy module paths remain aliases to the same modules.

```python
from CTRAIN.model_wrappers.certified import ShiIBPModelWrapper, PropertyModelWrapper
from CTRAIN.model_wrappers.adversarial import PGDModelWrapper
```

The robustness wrappers keep their `eps` constructors. General-property method
selection is configured on `PropertyModelWrapper`; see the
[property guide](../../properties.md#choosing-a-training-strategy).

`CTRAINWrapper.evaluate(...)` returns `(standard_accuracy, certified_accuracy,
adversarial_accuracy)`.

`CTRAINWrapper.evaluate_complete(...)` also returns the same aggregate tuple,
but additionally persists complete-verification details to
`results_path/results_filename` and writes abCROWN logs under
`results_path/abCROWN_logs`. Use `warm_start=True` to reuse existing results, or
`start_idx`/`end_idx` to verify a dataset slice. Complete verification expects
the pinned abCROWN-compatible `auto_LiRPA` installation provided by
`ctrain-install-git-deps`.

Use a sequential evaluation loader with `drop_last=False`. Complete verification
captures the evaluated inputs in CPU memory so screening and the solver use the
same transformed images. Each results file has a `.manifest.json` sidecar that
binds it to the exported model, images, labels, class count, radius and clipping
bounds. A warm start rejects mismatches and legacy files without a manifest;
rerun those with `warm_start=False`. Keep the sidecar alongside the results.

`CTRAINWrapper.hpo(...)` is the standard HPO entry point. It runs the
novel multi-objective optimisation procedure, maximizes natural and
certified validation accuracy, stores the Optuna study in
`output_dir/optuna_study.db`, writes trial checkpoints to `output_dir/nets`, and
returns the whole Pareto front. It does not load one checkpoint implicitly; pick
the Pareto point you want and load its `checkpoint_path`. If `min_nat_acc` or
`min_cert_acc` are set, the returned front is computed over feasible trials when
any feasible trial exists; otherwise, it falls back to the unconstrained front
and marks entries as infeasible.

`CTRAINWrapper.hpo_single_objective(...)` runs scalar Optuna HPO. By default it
maximizes `nat_acc + cert_acc`, returns the best trial record, and loads that
checkpoint unless `load_best=False`. Change the scalar objective with
`nat_acc_weight`, `adv_acc_weight`, and `cert_acc_weight`.

Both Optuna entry points default to BoTorch. CTRAIN declares
`optuna-integration[botorch]` as a dependency, which supplies the compatible
BoTorch version. Scalar HPO also accepts `sampler="tpe"` or `sampler="random"`.
Use a validation holdout for tuning and reserve the test split for final reporting.

`CTRAINWrapper.hpo_smac(...)` keeps the older SMAC3 single-objective workflow.
Use `sampler="botorch"` for the default constrained Bayesian optimization
sampler or `sampler="nsgaii"` for a lighter Optuna-only smoke test.

`resume_from_checkpoint(...)` loads the training model, active optimizer and LR
scheduler, and continues at the next epoch. Epoch checkpoints store scheduler
state after that epoch's LR update. Legacy checkpoints without scheduler state
cannot resume reliably and are rejected; their `model_state_dict` can still be
loaded for inference. RNG and loader state are not restored, so resumption does
not promise a bitwise replay. TAPS/STAPS may restore the best inference weights
after training; trajectory checkpoints remain separate from that selection.

::: CTRAIN.model_wrappers.certified.model_wrapper
