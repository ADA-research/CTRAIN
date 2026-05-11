`CTRAINWrapper.evaluate(...)` returns `(standard_accuracy, certified_accuracy,
adversarial_accuracy)`.

`CTRAINWrapper.evaluate_complete(...)` also returns the same aggregate tuple,
but additionally persists complete-verification details to
`results_path/results_filename` and writes abCROWN logs under
`results_path/abCROWN_logs`. Use `warm_start=True` to reuse existing results, or
`start_idx`/`end_idx` to verify a dataset slice. Complete verification expects
the newer abCROWN-compatible `auto_LiRPA` installation, such as
`auto_LiRPA 0.7.0`.

::: CTRAIN.model_wrappers.model_wrapper
