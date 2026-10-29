# Troubleshooting

Training and verification depend on a consistent model, input representation and numerical environment. When a run fails, first identify whether the failure occurs during imports, bound computation, model export or solver execution. The following sections explain the checks that help distinguish these cases.

## Dependency and Interpreter Mismatches

An import error can arise when a notebook kernel, training script and dependency installer use different Python environments. First, check the interpreter and the auto_LiRPA module actually being loaded:

```python
import sys
import auto_LiRPA

print("Python:", sys.executable)
print("auto_LiRPA:", auto_LiRPA.__file__)
```

Run CTRAIN and `ctrain-install-git-deps` from the intended environment, and select the same environment as the Jupyter kernel. The installer obtains CTRAIN's pinned Git dependencies. An older auto_LiRPA revision may lack interfaces required by the bundled abCROWN code, such as `stop_criterion_general`.

Complete-verification backends can depend on different forks or runtime conventions. Installing an upstream verifier's requirements over the training environment can replace the versions CTRAIN expects. Follow the [backend setup instructions](../complete_verification.md#backend-setup); NeuralSAT runs in a subprocess and can use a separate interpreter when configured accordingly.

For documentation builds, use the development dependencies from the [installation guide](../getting_started.md#installation). MkDocs, Material, mkdocstrings and notebook rendering must be compatible within the interpreter used for the build. Rendering errors do not establish a failure of the training or verification computation.

## Invalid Bounds or Training Metrics

If a loss or bound becomes non-finite, first check the input batch and its metadata. Inputs must be finite, channel standard deviations positive and finite, and clipping limits consistent with the coordinates consumed by the model. The model and input tensors must also agree in device and dtype. The [data guide](data_and_models.md#input-coordinates-and-loader-metadata) explains the required loader fields.

Next, check the configured radius and schedule. Warm-up, ramp-up and learning-rate milestones should fit the intended epoch budget. Classification radii are specified in raw pixel units, even when images are normalised. General-property boxes already use verification-model coordinates. Scaling either representation twice changes the problem being trained or verified.

An attack search factor changes where the attack searches; it does not redefine the nominal certification region. Similarly, ReLU shrinkage and selected subregions may be useful training surrogates, but their bounds do not certify the full input region. Assess the trained model with fresh [evaluation](evaluation.md).

Some training statistics are unavailable for particular computation graphs. For example, CROWN-IBP's fused loss graph does not expose the per-class margins needed for its certified training metric, which may be reported as `NaN`. This statistic is separate from the optimisation loss. Check that the loss and gradients are finite, then compute the desired accuracy through evaluation; see [CROWN-IBP loss fusion](../crown_ibp.md#loss-fusion).

A small synthetic run can confirm that a path executes. Training quality and published-score comparisons require the intended dataset, architecture, schedules and evaluation protocol.

## ONNX Parity Failures

CTRAIN checks exported ONNX predictions against PyTorch before using the graph for verification. The shared `export_onnx` helper uses `rtol=1e-4` and `atol=1e-5`, rejects invalid shapes and non-finite outputs, and raises when the numerical comparison cannot be established.

For a failed FP32 comparison, the helper also tries a separate CPU float64 copy on the same input values with unchanged tolerances. If this reference agrees, export proceeds with the original FP32 ONNX graph and emits a `RuntimeWarning` containing both errors. The caller's weights, dtype, device, modes and gradient flags are preserved. If the reference is unsupported or also disagrees, export fails.

When investigating a mismatch, first confirm the architecture and checkpoint, then the preprocessing and input coordinates. Compare representative clean and perturbed inputs in evaluation mode. FP32 accumulation order and TF32 settings can affect numerical agreement; moving a model to a different device alone does not demonstrate equivalence. Near-zero logits can fail relative tolerances even when predicted classes agree.

Preserve parity warnings with the experiment results. The float64 comparison is a numerical diagnostic on probes, and the ONNX graph being certified remains the exported graph. Neither successful probes nor matching class predictions establish uniform equivalence over an input box. Relaxing tolerances solely to obtain a solver result would conceal this discrepancy.

## Timeouts and Reproducibility

A complete-verification timeout leaves a query unresolved. It establishes neither safety nor a witnessed violation. Inspect the per-input result and backend log to distinguish a timeout from an invocation or export failure. The [complete-verification guide](../complete_verification.md) describes the result files and solver-specific controls.

Warm-start results require their identity manifest. Use a new results directory when changing solver settings or retrying cached timeouts, and retain input ordering when splitting an evaluation. See [evaluation continuation](evaluation.md#continuing-and-splitting-an-evaluation).

When resuming training, load a trajectory checkpoint rather than only inference weights. Validation selection may have restored an earlier model without rewinding the optimiser or scheduler. Classifier trajectories also omit RNG and loader state, so continuation need not reproduce the same stochastic updates. The [checkpoint guide](checkpoints.md) explains the states retained by each workflow.

For a reproducible report, keep the configuration, preprocessing, split, seed, training and selection policy, export warnings and verification budgets alongside the results. Investigation logs should remain associated with the experiment that produced them.
