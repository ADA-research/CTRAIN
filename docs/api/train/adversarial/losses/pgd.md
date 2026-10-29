# PGD training loss API

This loss supports native PyTorch classifiers and is used by the
[PGD trainer](../pgd.md). The attack runs in evaluation mode; adversarial loss
evaluation uses the caller's mode. It returns a scalar loss, or `(loss, error_rate)`
when `return_stats=True`. `return_bounds=True` is unsupported.

The step size is relative to the attack region. These are training controls;
see the [evaluation guide](../../../../guides/evaluation.md) for evaluation budgets.

::: CTRAIN.train.adversarial.losses.pgd.get_pgd_loss
