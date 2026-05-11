The CTRAIN abCROWN integration exports the model to ONNX, writes per-instance
VNNLIB specifications, and records complete-verification results as JSON. The
evaluation API supports resumable runs via `warm_start` and chunked verification
with `start_idx`/`end_idx`; see `CTRAIN.eval.eval_complete_abcrown` and
`CTRAIN.model_wrappers.CTRAINWrapper.evaluate_complete`.

The vendored verifier is updated to abCROWN commit
`6b8bbcfac1c01da1cabd240a87e4dce1a65f5a2b` and uses the matching newer default
configuration schema. Run complete verification in an environment with the
matching `auto_LiRPA` package installed; the training-experiments venv currently
provides `auto_LiRPA 0.7.0`.

::: CTRAIN.complete_verification.abCROWN.verify

::: CTRAIN.complete_verification.abCROWN.runner

::: CTRAIN.complete_verification.abCROWN.util
