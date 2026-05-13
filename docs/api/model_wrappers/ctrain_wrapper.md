`CTRAINWrapper.evaluate(...)` returns `(standard_accuracy, certified_accuracy,
adversarial_accuracy)`.

`CTRAINWrapper.evaluate_complete(...)` also returns the same aggregate tuple,
but additionally persists complete-verification details to
`results_path/results_filename` and writes abCROWN logs under
`results_path/abCROWN_logs`. Use `warm_start=True` to reuse existing results, or
`start_idx`/`end_idx` to verify a dataset slice. Complete verification expects
the newer abCROWN-compatible `auto_LiRPA` installation, such as
`auto_LiRPA 0.7.0`.

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

`CTRAINWrapper.hpo_smac(...)` keeps the older SMAC3 single-objective workflow.
Use `sampler="botorch"` for the default constrained Bayesian optimization
sampler or `sampler="nsgaii"` for a lighter Optuna-only smoke test.

::: CTRAIN.model_wrappers.model_wrapper
