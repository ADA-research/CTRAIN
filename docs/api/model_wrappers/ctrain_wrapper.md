# Model wrappers

Certified and hybrid implementations live in `CTRAIN.model_wrappers.certified`;
native PGD training lives in `CTRAIN.model_wrappers.adversarial`. Existing package
exports and legacy direct module paths remain compatible aliases.

The user-facing contracts are documented once in the guides:

- [Training options and modes](../../guides/training.md).
- [Evaluation and complete-verification results](../../guides/evaluation.md).
- [Checkpoint formats, resumption and validation selection](../../guides/checkpoints.md).
- [Optuna/SMAC HPO and Pareto selection](../../guides/hpo.md).
- [General-property workflow](../../properties.md).

Advanced HPO integrations can inspect the [configuration-space API](configs.md).

::: CTRAIN.model_wrappers.certified.model_wrapper
