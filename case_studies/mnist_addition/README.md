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
also available. The full default budget is 100 trials per seed and 100 epochs.

HPO tunes learning rate, property weight, weight ramp, property batch size and
training box count. MTL-IBP adds `alpha` in [0, 1]; SABR adds `subselection_ratio`
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
