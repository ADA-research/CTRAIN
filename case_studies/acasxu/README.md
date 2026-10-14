# ACAS Xu repair

Repair an existing five-input, five-advisory network while preserving its original
`argmin` advisories with cross-entropy on negated scores. Affine/Boolean output
properties are imported from VNNLIB. Its usual unsafe predicate is complemented
to obtain the desired safe property; ties and strict comparisons are preserved.
Normalization is not added by the loader. ONNX conversion is checked against
ONNX Runtime at deterministic probes, without claiming graph equivalence.

See [setup](../README.md) and the [notebook](../../docs/examples/case_studies/acasxu.ipynb).

## Small execution check and one repair

```bash
python -m case_studies.acasxu.run --smoke --device cpu --output runs/case_studies/acasxu_smoke
python -m case_studies.acasxu.run --model data/acasxu/onnx/ACASXU_run2a_2_9_batch_2000.onnx --specification data/acasxu/vnnlib/prop_8.vnnlib --epochs 40 --output runs/case_studies/acasxu_2_9
```

Use `--additional-specifications FILE ...` to train assigned properties jointly.
`--input-shape` resolves dynamic ONNX dimensions; `--output-semantics safe` is
only for files already expressing the desired property. `--method` and
`--training-options` select the general-property training strategy.

The two released presets (properties 7 and 8) can be downloaded explicitly from
VNN-COMP 2021 using an immutable, full 40-character revision:

```bash
python -m case_studies.acasxu.run --artifacts data/acasxu_presets --download --revision YOUR_FULL_COMMIT_SHA --case 8 --output runs/case_studies/acasxu_preset8
```

This preset download does not acquire the complete benchmark. For the full
study, supply a local benchmark directory with `instances.csv`, `onnx/` and
`vnnlib/`, retaining its upstream revision and license.

## Joint repair, evaluation and HPO

```bash
python -m case_studies.acasxu.suite --benchmark data/acasxu --output runs/case_studies/acasxu_suite --dry-run
python -m case_studies.acasxu.evaluate --run runs/case_studies/acasxu_2_9 --benchmark data/acasxu --assigned-only --method CROWN --output runs/case_studies/acasxu_2_9_comparison
python -m case_studies.acasxu.hpo --benchmark data/acasxu --network 2_9 --training-method MTL-IBP --trials 20 --epochs 40 --output runs/case_studies/acasxu_hpo/2_9
```

Remove `--dry-run` to execute the suite. It repairs each selected network once,
including all its assigned properties, and compares CROWN-IBP/CROWN bounds.
`--evaluation-methods CROWN-IBP CROWN ALPHA-CROWN` also optimizes relaxation
slopes. The suite comparison uses incomplete verification only.

HPO creates a separate study per network, using 4,096 fidelity inputs, independent
selection/test streams and 32,768 probes per source box. It tunes learning rate,
property weight, hinge margin, weight ramp and property batch size. MTL-IBP also
tunes `alpha` in [0, 1]; SABR tunes the coordinatewise half-width ratio
`subselection_ratio` logarithmically in [0.01, 1]. Other method options stay fixed.
The search is Optuna TPE with ten startup trials, three enqueued configurations
and final-epoch weights. These are exploratory budgets, not optimality claims.

Selection prefers complete certification on the fixed 64-cell CROWN cover,
then worst sampled satisfaction, then the mean of property-box and envelope
advisory agreement. The selected checkpoint receives fresh whole-domain CROWN
and abCROWN checks. Better-fidelity trials that fail the fixed cover are not
fully verified automatically. Keep this selection limitation in mind.

For a portable full-benchmark collection, copy the benchmark's `instances.csv`,
`onnx/` and `vnnlib/` into your fresh study root before running network studies
there, so `--collect` can reconstruct the assignments. It also accepts an
existing suite `submission.json` with a `cases` list.

```bash
python -m case_studies.acasxu.hpo --collect runs/case_studies/acasxu_hpo
```

`original_certified` preserves an already certified original. `repaired_certified`
requires every assigned obligation to be certified; unknown or violated selected
models remain `repair_incomplete`. Sampling and advisory agreement are empirical.
Floating-point LiRPA/abCROWN checks are not directed-rounding proofs.
