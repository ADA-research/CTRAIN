# Evaluation API

See [evaluation and verification](../../guides/evaluation.md) for metric meanings,
normalization, attack budgets, complete-verifier records and warm-start identity.
The [evaluation notebook](../../examples/model_evaluation.ipynb) is the runnable workflow.

Forward LiRPA certification uses `eval_forward` or
`eval_certified(method="FORWARD")`; `eval_model` combines clean, certified and
adversarial accuracy. Radii passed directly to these functions are in model
input coordinates; wrappers convert raw-pixel radii using loader metadata.
`propagation_batch_size` limits forward-bound memory; it defaults to one.

See also [general-property certification/counterexample search](properties.md).

::: CTRAIN.eval.eval

::: CTRAIN.eval.autoattack.eval_autoattack
