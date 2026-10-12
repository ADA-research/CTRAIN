Evaluation helpers return aggregate accuracies and, for the lower-level
certification routines, per-instance result dictionaries. A result entry uses
`result="unsat"` for certified samples, `result="sat"` for adversarial
counterexamples, and `result=None` for unresolved samples. The `method` and
`running_time` fields record which verifier produced the result and how long it
took.

`eval_model` and wrapper `evaluate(...)` keep the stable public return shape:
`(standard_accuracy, certified_accuracy, adversarial_accuracy)`.

Alpha-CROWN optimizes ReLU slopes using auto_LiRPA's `CROWN-Optimized` method.
Use `eval_alpha_crown(...)`, `eval_certified(..., method="ALPHA-CROWN")`, or
include `"ALPHA-CROWN"` in an adaptive method list. `alpha_iterations` defaults
to 20. `CTRAIN.bound.bound_alpha_crown(...)` exposes the bound computation.
Evaluation restores caller modes, parameter gradient flags and optimization
options, including on failure.

AutoAttack uses the optional [official implementation](https://github.com/fra31/auto-attack).
Install it with `ctrain-install-git-deps`, which installs the pinned upstream
revision alongside CTRAIN's other Git-hosted dependencies.
`eval_autoattack(...)` and `eval_adversarial(..., method="AUTOATTACK")` return
empirical adversarial accuracy, not a certificate. With `return_adv_indices=True`
they return `(accuracy, failure_mask, per_instance_results)` like the PGD evaluator.
The default is the standard Linf suite. `version`, `seed` and `verbose` are
available on the standalone function or through `autoattack_kwargs` in the
combined evaluator. Inputs are converted from loader coordinates to pixel space
using loader `min`/`max`, then normalized inside the attacked model. The supplied
epsilon must describe a uniform scalar pixel-space radius. Dataset batches are
attacked independently; batch partitioning can affect randomized attack results.

```python
clean, certified, adversarial = wrapper.evaluate(
    test_loader,
    eval_method="ALPHA-CROWN",
    alpha_iterations=50,
    adversarial_method="AUTOATTACK",
    autoattack_kwargs={"version": "standard", "seed": 0},
)
```

Existing calls retain PGD evaluation by default. Certification and adversarial
methods are chosen separately. Set-based wrappers also support this selection,
including AutoAttack with their default zonotope certification.

Complete verification through `eval_complete_abcrown` writes a JSON result file
and per-instance abCROWN logs under `results_path`. It supports `warm_start`,
`start_idx`, `end_idx`, and `results_filename` so large evaluations can be
resumed or split into chunks. The abCROWN backend is the newer vendored checkout
and should be run with a matching `auto_LiRPA` installation, for example
`auto_LiRPA 0.7.0`.

::: CTRAIN.eval.eval

::: CTRAIN.eval.autoattack.eval_autoattack
