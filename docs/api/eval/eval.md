Evaluation helpers return aggregate accuracies and, for the lower-level
certification routines, per-instance result dictionaries. A result entry uses
`result="unsat"` for certified samples, `result="sat"` for adversarial
counterexamples, and `result=None` for unresolved samples. The `method` and
`running_time` fields record which verifier produced the result and how long it
took.

`eval_model` and wrapper `evaluate(...)` keep the stable public return shape:
`(standard_accuracy, certified_accuracy, adversarial_accuracy)`.

Complete verification through `eval_complete_abcrown` writes a JSON result file
and per-instance abCROWN logs under `results_path`. It supports `warm_start`,
`start_idx`, `end_idx`, and `results_filename` so large evaluations can be
resumed or split into chunks. The abCROWN backend is the newer vendored checkout
and should be run with a matching `auto_LiRPA` installation, for example
`auto_LiRPA 0.7.0`.

::: CTRAIN.eval.eval
