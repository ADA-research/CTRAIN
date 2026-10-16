# MNIST addition consistency

The network receives two normalized digit images and predicts three heads:
10 logits for each operand and 19 for the sum. Cross-entropy supervises integer
labels [a, b, a+b]. The property requires the predicted sum to equal the sum of
the predicted operands throughout a clipped raw-pixel perturbation box.
Consistency can hold even when all three predictions are wrong.

See [setup](../README.md) and the [notebook](../../docs/examples/case_studies/mnist_addition.ipynb).

## Paired baselines

```bash
python -m case_studies.mnist_addition.run --smoke --device cpu --output runs/case_studies/addition_smoke
python -m case_studies.mnist_addition.run --config case_studies/mnist_addition/config.json --download --device cpu --output runs/case_studies/addition_real
```

The real run explicitly downloads missing MNIST into `--data-root` (default
`data/`). Both arms share initialization and seeded batches: CE-only versus
CE plus property loss. Training pairs cover every source training image in
each operand position. Raw pixels are clipped before normalization; evaluation
is unaugmented. The JSON config uses CPU by default; select CUDA explicitly
for full training. CLI flags override configuration. Synthetic smoke metrics
are execution checks, not MNIST evidence.

Add `--skip-ce-baseline` to train only the model with property loss. The
`mnist_playground.py` launcher trains both models by default; set
`"skip-ce-baseline": True` there to skip the baseline. In swap mode both models
receive the same accuracy, empirical swap-agreement and certification evaluation.
The baseline trains only on CE; the other model adds the property penalty.
They share initialization, training pairs and task-batch order. Results are saved
under `ce_only/` and `certified/`, with their comparison in the run's `results.json`.

Every training pair defines a property box. Boxes are generated in memory on
demand, and each task update samples `--property-batch-size` boxes, uniformly
with replacement from the entire training-pair pool, independently of its task
batch. `--batch-size` controls task examples per update independently. Even a
partial final task batch samples the configured number of property domains.
`--property-batch-size` also controls batches during full-pool certification.
Sampling does not guarantee that every box contributes to the penalty in an
epoch. In the standalone runner/playground, `--property-boxes N` limits the property pool to the first N
seeded training pairs. Task training still covers all pairs. Omitting it uses
all pairs. HPO uses the configured property pool and does not tune this count.
The shared arithmetic formula is constructed directly; these runs write no
VNNLIB files. The optional `write_vnnlib` helper remains available for export.

After each epoch following warm-up, fresh bounds evaluate every training box in batches, as in
the general-property workflow. This full scan is substantially more expensive
than checking the former small subset. Training certification reports contain
aggregate counts and margins; validation/test reports retain individual boxes.
Checkpoints record a dataset hash, box-generation settings and random state;
reconstruct the same training pairs when resuming. Lazy datasets support
full-box bounds; refinement and complete verification use explicit specifications.

Use `--warm-up-epochs N` for an initial task-only phase and
`--weight-ramp-epochs R` to increase the property weight over the next R epochs.
Both phases count toward `--epochs`; the remaining epochs use full weight.
Warm-up skips property sampling and certification scans. The supplied config
uses zero warm-up and a 15-epoch ramp within its 50-epoch budget.

The property wrapper shows a tqdm task-batch progress bar and logs the full-pool
certified count/fraction after each active epoch. Warm-up is labelled as not
evaluated. MNIST also supplies per-head and triple classification accuracy,
plus per-head and overall empirical swap agreement in swap mode. Standalone
runs use test pairs for these diagnostics; HPO uses selection pairs and keeps
test evaluation until after selection. Regression supplies MSE/MAE and empirical
property satisfaction. Diagnostics do not select checkpoints. The wrapper saves
epoch history to each arm/trial's `checkpoints/history.json` after every epoch,
so partial results are available while training is running.

## Swap equivariance at exact pairs

Set `"property": "swap"` and keep `"task": "classification"` in
`mnist_playground.py`, or run:

```bash
python -m case_studies.mnist_addition.run --property swap \
  --skip-ce-baseline --device cuda --epochs 50 --batch-size 128 \
  --output runs/case_studies/addition_swap
```

This replaces arithmetic consistency with agreement under operand exchange:
the original operand predictions must match the opposite heads after swapping,
and the predicted sum must stay the same. It compares winning classes, not
logits. Task training retains the three cross-entropy heads.

The verification graph concatenates `f(x)` and `f(swap(x))`, sharing all model
weights. The swap is a fixed channel permutation inside the graph, so both
evaluations depend on the same input. Each property domain is an exact
normalized training pair, represented as `Box(x, x)`; swap mode sets the
effective `epsilon` to zero. Sampling, the chosen property training method,
warm-up, ramp and adjustable property pool work as before. No VNNLIB files
are needed. Reports include empirical class agreement and bound certificates
for train/test pairs. These certificates cover those pairs only; argmax ties
can remain unresolved because certification requires positive margins.
Regression retains its arithmetic property. HPO supports `--property swap`.

## Real-valued regression variant

Set `"task": "regression"` in `mnist_playground.py`, or run:

```bash
python -m case_studies.mnist_addition.run --task regression --tau 0.1 \
  --skip-ce-baseline --device cuda --epochs 50 --batch-size 128 \
  --property-boxes 1024 --output runs/case_studies/addition_regression
```

The existing regression MLP independently estimates `[a, b, sum]` as three
unconstrained real values. Mean squared error supervises `[a, b, a+b]`.
The property is exactly `sum - a - b <= tau` throughout each pixel box.
`tau` is a finite nonnegative residual tolerance; it is separate from the
hinge slack and SABR's subregion ratio. This is one-sided: arbitrarily negative
residuals satisfy it. It does not enforce an absolute residual tolerance or
correct digit estimates. Reports include per-output MSE/MAE, signed residuals,
sampled satisfaction and certified satisfaction. Without the skip flag, the
baseline is saved as `task_only`. HPO and the prediction-stability diagnostic
remain classification-only.

## HPO and comparisons

```bash
python -m case_studies.mnist_addition.hpo --method MTL-IBP --trials 20 --output runs/case_studies/addition_hpo/seed_0 --dry-run
python -m case_studies.mnist_addition.hpo --method MTL-IBP --trials 20 --device cuda --output runs/case_studies/addition_hpo/seed_0
```

Use the baseline command with `--download` first if MNIST is not cached. Repeat
with `--seed 1` and `--seed 2` in fresh `seed_1`/`seed_2` directories, keeping
all other options the same, then pool the selection fronts:

```bash
python -m case_studies.mnist_addition.hpo --summarize runs/case_studies/addition_hpo
```

The default protocol holds out 6,000 training images before pairing, uses
1,000 selection pairs and 1,000 official-test pairs, and does not use test scores
for selection. `--selection-split test` explicitly changes that protocol.
BoTorch is the default sampler with ten startup trials; `--sampler nsgaii` is
also available. The full default budget is 100 trials per seed and 50 epochs.

HPO tunes learning rate, property weight, warm-up and weight ramp. Task and
property batch sizes are fixed independently by the config or `--batch-size`
and `--property-batch-size`; neither is a search dimension.
For 50 epochs, warm-up is searched over 0–5 epochs and the subsequent weight
ramp over 0–25 epochs. Shorter budgets reduce the ranges so both phases fit.
Every trial uses the configured training-property pool (`property_boxes` in the
JSON config; default: all training pairs). MTL-IBP adds `alpha` in
[0, 1]; SABR adds `subselection_ratio`
in [0.01, 1], sampled logarithmically. Baselines keep their configured options;
all other method parameters remain fixed. `--ce-only` tunes only learning rate.
A matched CE-only root can be supplied to `--summarize` with `--ce-study ROOT`.
`--allow-partial --output FRESH_DIRECTORY` exports an interim comparison without
altering running studies. JSON, CSVs and plots retain the selection protocol,
trial options, source hashes and checkpoint locations.

## Prediction-stability diagnostic

Choose a Pareto-selected trial with saved `test_verification.json`:

```bash
python -m case_studies.mnist_addition.local_robustness --study-root runs/case_studies/addition_hpo --seed 0 --trial 12 --data-root data --device cpu --output runs/case_studies/addition_stability
python -m case_studies.mnist_addition.local_robustness --self-test
```

Replace trial 12 with your saved trial number. The diagnostic recomputes CROWN
margins for fixed winning triples and searches for changed predictions. It
checks the cached dataset hashes against saved test provenance. `--limit N`
checks only a prefix and is not a full evaluation. Unsuccessful attacks and
failed bounds remain unresolved. These formulas and the current affine-leaf
certifier can certify a fixed consistent winning triple; they do not establish
flexibly changing, consistently correct digit recognition.
