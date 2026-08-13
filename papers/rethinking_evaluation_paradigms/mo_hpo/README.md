# Multi-Objective HPO Reproduction

This directory contains the CTRAIN-side reproduction entrypoints for the paper
preprint "Rethinking Evaluation Paradigms in IBP-based Certified Training".
The paper argues that IBP-based certified training methods should be compared
by their Pareto fronts over natural and certified accuracy instead of by a
single tuned configuration. The implemented workflow is:

1. Run constrained multi-objective HPO for each dataset, architecture, method,
   radius, and seed.
2. Combine the three seed-wise Optuna studies into one Pareto front.
3. Subselect non-redundant Pareto configurations before expensive complete
   verification.
4. Run complete verification on the selected checkpoints.
5. Generate plots, tables, hypervolume/convergence summaries, and
   hyperparameter analyses.

The HPO objective uses natural and certified accuracy from incomplete
verification on the loader selected by `run_hpo.py`. Without `--val-split`,
that loader is the test set, matching the intended publication protocol and
prior certified-training comparisons. `--val-split` is an optional alternate
protocol used by the validation-vs-test tuning analysis.

## Setup

Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ctrain-install-git-deps
```

The default HPO sampler is BoTorch through Optuna, so the environment must have
`optuna`, `optuna-integration[botorch]`, `botorch`, and the CTRAIN verification
dependencies installed. These are included in the package metadata.

For a quick local smoke test without BoTorch, run the HPO documentation example
and use the `sampler="nsgaii"` cell:

```bash
jupyter notebook docs/examples/hyperparameter_optimisation.ipynb
```

## One HPO Job

This is the basic command used by all reproduction loops below:

```bash
python papers/rethinking_evaluation_paradigms/mo_hpo/run_hpo.py \
  --dataset cifar10 \
  --network cnn7 \
  --method mtl_ibp \
  --eps 0.00784313725490196 \
  --seed 0 \
  --epochs 160 \
  --budget-trials 100 \
  --min-cert-acc 0.40 \
  --min-nat-acc 0.60 \
  --output-root papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies
```

Each run writes:

- Optuna study: `papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies/{dataset}_{network}_{method}_{eps}_{seed}/optuna_study.db`
- Checkpoints: `papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies/{dataset}_{network}_{method}_{eps}_{seed}/nets/{config_hash}.pt`

Use `--sampler nsgaii` for dependency-light tests. Use the default
`--sampler botorch` for paper-style runs.

## MTL-IBP Alpha Sweep

To isolate the MTL-IBP alpha trade-off, run the fixed-alpha sweep for CIFAR-10.
The script only takes the epsilon on the command line; edit the constants at the
top of the script to change seeds, alpha grid, architecture, training settings,
or output path. The per-epsilon training defaults are copied from
`/storage/work/kaulen/CTRAIN_HPO/defaults/cifar10_2_255/mtl_ibp.py` and
`/storage/work/kaulen/CTRAIN_HPO/defaults/cifar10_8_255/mtl_ibp.py`.

```bash
python papers/rethinking_evaluation_paradigms/mo_hpo/run_mtl_ibp_alpha_sweep.py 2/255
python papers/rethinking_evaluation_paradigms/mo_hpo/run_mtl_ibp_alpha_sweep.py 8/255
```

Submit both epsilons as a SLURM array:

```bash
sbatch papers/rethinking_evaluation_paradigms/mo_hpo/submit_mtl_ibp_alpha_sweep.slurm
```

The default alpha grid contains about 20 values: the original MTL-IBP alpha for
the selected epsilon, plus logarithmic deviations below it down to `1e-6` and
above it up to `1`. The script uses one seed by default. Each epsilon directory
contains `alpha_sweep.csv`, `pareto_front.csv`, saved checkpoints under `nets/`,
and `summary.json` with the Pareto-front hypervolume. Hypervolume is computed
over natural and certified accuracy with reference point `(0, 0)` by default.

## Main Paper HPO Runs

The paper uses 100 HPO trials per seed and three seeds, yielding 300 trials per
benchmark. Main-paper benchmarks use CNN7 on CIFAR-10 at `2/255` and `8/255`
and Tiny ImageNet at `1/255`.

CIFAR-10, `eps=2/255`, CNN7, 160 epochs:

```bash
for method in mtl_ibp sabr shi crown_ibp_nofusion; do
  for seed in 0 1 2; do
    python papers/rethinking_evaluation_paradigms/mo_hpo/run_hpo.py \
      --dataset cifar10 \
      --network cnn7 \
      --method "${method}" \
      --eps 0.00784313725490196 \
      --seed "${seed}" \
      --epochs 160 \
      --budget-trials 100 \
      --min-cert-acc 0.40 \
      --min-nat-acc 0.60 \
      --output-root papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies
  done
done
```

CIFAR-10, `eps=8/255`, CNN7, 260 epochs:

```bash
for method in mtl_ibp sabr shi crown_ibp_nofusion; do
  for seed in 0 1 2; do
    python papers/rethinking_evaluation_paradigms/mo_hpo/run_hpo.py \
      --dataset cifar10 \
      --network cnn7 \
      --method "${method}" \
      --eps 0.03137254901960784 \
      --seed "${seed}" \
      --epochs 260 \
      --budget-trials 100 \
      --min-cert-acc 0.25 \
      --min-nat-acc 0.40 \
      --output-root papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies
  done
done
```

Tiny ImageNet, `eps=1/255`, CNN7, 160 epochs:

```bash
for method in mtl_ibp sabr shi crown_ibp; do
  for seed in 0 1 2; do
    python papers/rethinking_evaluation_paradigms/mo_hpo/run_hpo.py \
      --dataset tinyimagenet \
      --network cnn7 \
      --method "${method}" \
      --eps 0.00392156862745098 \
      --seed "${seed}" \
      --epochs 160 \
      --budget-trials 100 \
      --min-cert-acc 0.15 \
      --min-nat-acc 0.20 \
      --output-root papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies
  done
done
```

## Appendix HPO Runs

MNIST, `eps=0.3`, CNN7, 70 epochs, 300s complete-verification cutoff in the
paper analysis:

```bash
for method in mtl_ibp sabr shi crown_ibp_nofusion; do
  for seed in 0 1 2; do
    python papers/rethinking_evaluation_paradigms/mo_hpo/run_hpo.py \
      --dataset mnist \
      --network cnn7 \
      --method "${method}" \
      --eps 0.3 \
      --seed "${seed}" \
      --epochs 70 \
      --budget-trials 100 \
      --min-cert-acc 0.90 \
      --min-nat-acc 0.95 \
      --output-root papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies
  done
done
```

CIFAR-10 architecture study, `eps=2/255`, 160 epochs. The preprint reports
CNN5, CNN7, CNN7 Wide, CNN7 Narrow, and CNN9; CNN3/CNN11 are supported by the
runner for compatibility with the publication fork but were not part of the
reported architecture figure.

```bash
for network in cnn5 cnn7 wide_cnn7 narrow_cnn7 cnn9; do
  for method in mtl_ibp sabr shi crown_ibp_nofusion; do
    for seed in 0 1 2; do
      python papers/rethinking_evaluation_paradigms/mo_hpo/run_hpo.py \
        --dataset cifar10 \
        --network "${network}" \
        --method "${method}" \
        --eps 0.00784313725490196 \
        --seed "${seed}" \
        --epochs 160 \
        --budget-trials 100 \
        --min-cert-acc 0.40 \
        --min-nat-acc 0.60 \
        --output-root papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies
    done
  done
done
```

CIFAR-10 ResNet18 runs from the publication grid:

```bash
for eps_epochs_thresholds in \
  "0.00784313725490196 160 0.40 0.60" \
  "0.03137254901960784 260 0.25 0.40"; do
  set -- ${eps_epochs_thresholds}
  eps="$1"; epochs="$2"; min_cert="$3"; min_nat="$4"
  for method in mtl_ibp sabr shi crown_ibp_nofusion; do
    for seed in 0 1 2; do
      python papers/rethinking_evaluation_paradigms/mo_hpo/run_hpo.py \
        --dataset cifar10 \
        --network resnet18 \
        --method "${method}" \
        --eps "${eps}" \
        --seed "${seed}" \
        --epochs "${epochs}" \
        --budget-trials 100 \
        --min-cert-acc "${min_cert}" \
        --min-nat-acc "${min_nat}" \
        --output-root papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies
    done
  done
done
```

Tiny ImageNet architecture runs from the publication grid:

```bash
for network in cnn7 wide_cnn7 resnet18; do
  for method in mtl_ibp sabr shi crown_ibp; do
    for seed in 0 1 2; do
      python papers/rethinking_evaluation_paradigms/mo_hpo/run_hpo.py \
        --dataset tinyimagenet \
        --network "${network}" \
        --method "${method}" \
        --eps 0.00392156862745098 \
        --seed "${seed}" \
        --epochs 160 \
        --budget-trials 100 \
        --min-cert-acc 0.15 \
        --min-nat-acc 0.20 \
        --output-root papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies
    done
  done
done
```

Validation-split variants use the same settings but pass `--val-split` and a
different output root. Example for the Tiny ImageNet validation-split run from
the publication repository:

```bash
for method in crown_ibp; do
  for seed in 0 1 2; do
    python papers/rethinking_evaluation_paradigms/mo_hpo/run_hpo.py \
      --dataset tinyimagenet \
      --network cnn7 \
      --method "${method}" \
      --eps 0.00392156862745098 \
      --seed "${seed}" \
      --epochs 160 \
      --budget-trials 100 \
      --min-cert-acc 0.15 \
      --min-nat-acc 0.20 \
      --val-split \
      --output-root papers/rethinking_evaluation_paradigms/results/hpo/validation/optuna_studies
  done
done
```

## Combine Seed-Wise Fronts

After the three HPO seeds finish, combine each group of studies. Example for
CIFAR-10, CNN7, MTL-IBP, `eps=2/255`:

```bash
python papers/rethinking_evaluation_paradigms/mo_hpo/calculate_fronts.py \
  --study papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies/cifar10_cnn7_mtl_ibp_0.00784313725490196_0/optuna_study.db \
  --study papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies/cifar10_cnn7_mtl_ibp_0.00784313725490196_1/optuna_study.db \
  --study papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies/cifar10_cnn7_mtl_ibp_0.00784313725490196_2/optuna_study.db \
  --method mtl_ibp \
  --eps 0.00784313725490196 \
  --output papers/rethinking_evaluation_paradigms/results/hpo/main/pareto_fronts/pareto_front_mtl_ibp_cnn7_cifar10_0.00784313725490196.csv
```

Batch command for the main-paper fronts:

```bash
mkdir -p papers/rethinking_evaluation_paradigms/results/hpo/main/pareto_fronts

for spec in \
  "cifar10 cnn7 0.00784313725490196 2_255 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 cnn7 0.03137254901960784 8_255 mtl_ibp sabr shi crown_ibp_nofusion" \
  "tinyimagenet cnn7 0.00392156862745098 1_255 mtl_ibp sabr shi crown_ibp"; do
  set -- ${spec}
  dataset="$1"; network="$2"; eps="$3"; eps_tag="$4"; shift 4
  for method in "$@"; do
    python papers/rethinking_evaluation_paradigms/mo_hpo/calculate_fronts.py \
      --study "papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies/${dataset}_${network}_${method}_${eps}_0/optuna_study.db" \
      --study "papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies/${dataset}_${network}_${method}_${eps}_1/optuna_study.db" \
      --study "papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies/${dataset}_${network}_${method}_${eps}_2/optuna_study.db" \
      --method "${method}" --eps "${eps}" \
      --output "papers/rethinking_evaluation_paradigms/results/hpo/main/pareto_fronts/pareto_front_${method}_${network}_${dataset}_${eps}.csv"
  done
done
```

Batch command for MNIST and the architecture appendix:

```bash
for spec in \
  "mnist cnn7 0.3 0_3 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 cnn5 0.00784313725490196 2_255 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 cnn7 0.00784313725490196 2_255 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 wide_cnn7 0.00784313725490196 2_255 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 narrow_cnn7 0.00784313725490196 2_255 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 cnn9 0.00784313725490196 2_255 mtl_ibp sabr shi crown_ibp_nofusion"; do
  set -- ${spec}
  dataset="$1"; network="$2"; eps="$3"; eps_tag="$4"; shift 4
  for method in "$@"; do
    python papers/rethinking_evaluation_paradigms/mo_hpo/calculate_fronts.py \
      --study "papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies/${dataset}_${network}_${method}_${eps}_0/optuna_study.db" \
      --study "papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies/${dataset}_${network}_${method}_${eps}_1/optuna_study.db" \
      --study "papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies/${dataset}_${network}_${method}_${eps}_2/optuna_study.db" \
      --method "${method}" --eps "${eps}" \
      --output "papers/rethinking_evaluation_paradigms/results/hpo/main/pareto_fronts/pareto_front_${method}_${network}_${dataset}_${eps}.csv"
  done
done
```

Each CSV contains the incomplete-verification objectives, portable source-study
directory, checkpoint `config_hash`, hyperparameters, and a deterministic
`subselected` flag implementing the paper's `0.05` objective-space clustering.
It is the preferred portable front format. The verification launchers also
accept the main paper's exact `_subselected0.05.txt` artifacts.

The repository includes the 24 complete CIFAR-10 validation-split Optuna
studies under `results/hpo/validation/optuna_studies/`. Pass their three database paths through
repeated `--study` arguments in exactly the same way. For example:

```bash
python papers/rethinking_evaluation_paradigms/mo_hpo/calculate_fronts.py \
  --study papers/rethinking_evaluation_paradigms/results/hpo/validation/optuna_studies/cifar10_cnn7_mtl_ibp_0.00784313725490196_0_complete_False/optuna_study.db \
  --study papers/rethinking_evaluation_paradigms/results/hpo/validation/optuna_studies/cifar10_cnn7_mtl_ibp_0.00784313725490196_1_complete_False/optuna_study.db \
  --study papers/rethinking_evaluation_paradigms/results/hpo/validation/optuna_studies/cifar10_cnn7_mtl_ibp_0.00784313725490196_2_complete_False/optuna_study.db \
  --method mtl_ibp --eps 0.00784313725490196 \
  --output papers/rethinking_evaluation_paradigms/results/hpo/validation/pareto_fronts/pareto_front_mtl_ibp_cnn7_cifar10_0.00784313725490196.csv
```

## Validation-Split Tuning Analysis

To compare the included CIFAR-10 validation-split studies against test-set
tuned CIFAR-10 CNN7 studies using incomplete verification, run from the
repository root:

```bash
python papers/rethinking_evaluation_paradigms/eval/validation_vs_test_tuning.py \
  --validation-root papers/rethinking_evaluation_paradigms/results/hpo/validation/optuna_studies \
  --test-root papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies
```

This writes Pareto-dominance and hypervolume summaries to
`papers/rethinking_evaluation_paradigms/results/analysis/validation_vs_test/incomplete/cifar10/`
and one validation-vs-test Pareto-front plot per matched setting under
`papers/rethinking_evaluation_paradigms/plots/validation_vs_test/incomplete/`.
Only feasible Optuna configurations, i.e. trials whose stored constraint values
are all non-positive, are included in the Pareto fronts, hypervolume, and plots;
pass `--include-infeasible` to reproduce the unfiltered diagnostic view.
By default the analysis caps each seed-wise Optuna study at the first 100
trials to match the publication budget; pass `--max-trials-per-study 0` to use
every completed trial in each database.
Once both tuning regimes are completely verified, compare their summaries
with:

```bash
python papers/rethinking_evaluation_paradigms/eval/validation_vs_test_tuning.py \
  --skip-incomplete \
  --validation-complete-summary papers/rethinking_evaluation_paradigms/results/verification/validation/summary_results_val_set.json \
  --test-complete-summary papers/rethinking_evaluation_paradigms/results/verification/main/summary_results.json \
  --output-dir papers/rethinking_evaluation_paradigms/results/analysis/validation_vs_test/complete/cifar10 \
  --plot-formats pdf,png
```

The parent [`README.md`](../README.md#validation-split-result-plots-optional)
contains the full audit, clean-accuracy, summary, and plotting sequence.

The optional expressive-loss analysis for CC-IBP and Exp-IBP uses the same
shared Optuna reader and Pareto rules:

```bash
python papers/rethinking_evaluation_paradigms/eval/plot_expressive_losses_results.py \
  --expressive-losses-root /path/to/expressive_losses_results \
  --mtl-root /storage/work/robust_nas/mosmac_ctrain \
  --plot-formats pdf
```

It writes aggregate CSVs under `results/analysis/expressive_losses/` and PDFs
under `plots/expressive_losses/`. The input roots can instead be set
with `CTRAIN_EXPRESSIVE_LOSSES_HPO_ROOT` and `CTRAIN_TEST_HPO_ROOT`.

## Complete Verification

The paper uses complete verification only after Pareto filtering/subselection:

- Main CNN7 CIFAR-10 and Tiny ImageNet: alpha-beta-CROWN timeout `1000s`.
- MNIST and architecture appendix: timeout `300s`.
- Architecture and MNIST appendix summary: the first `1000` test samples.
  Direct comparison with the archived summary confirms this selection.

Verify one combined front serially:

```bash
python papers/rethinking_evaluation_paradigms/mo_hpo/verify_front.py \
  --front papers/rethinking_evaluation_paradigms/results/hpo/main/pareto_fronts/pareto_front_mtl_ibp_cnn7_cifar10_0.00784313725490196.csv \
  --study-root papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies \
  --dataset cifar10 \
  --network cnn7 \
  --method mtl_ibp \
  --eps 0.00784313725490196 \
  --timeout 1000 \
  --test-samples 10000 \
  --abcrown-batch-size 1024 \
  --results-root papers/rethinking_evaluation_paradigms/results/verification/main
```

Batch command for the main-paper fronts:

```bash
for spec in \
  "cifar10 cnn7 0.00784313725490196 2_255 1000 10000 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 cnn7 0.03137254901960784 8_255 1000 10000 mtl_ibp sabr shi crown_ibp_nofusion" \
  "tinyimagenet cnn7 0.00392156862745098 1_255 1000 10000 mtl_ibp sabr shi crown_ibp"; do
  set -- ${spec}
  dataset="$1"; network="$2"; eps="$3"; eps_tag="$4"; timeout="$5"; samples="$6"; shift 6
  for method in "$@"; do
    python papers/rethinking_evaluation_paradigms/mo_hpo/verify_front.py \
      --front "papers/rethinking_evaluation_paradigms/results/hpo/main/pareto_fronts/pareto_front_${method}_${network}_${dataset}_${eps}.csv" \
      --study-root papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies \
      --dataset "${dataset}" \
      --network "${network}" \
      --method "${method}" \
      --eps "${eps}" \
      --timeout "${timeout}" \
      --test-samples "${samples}" \
      --results-root papers/rethinking_evaluation_paradigms/results/verification/main
  done
done
```

Batch command for MNIST and architecture-appendix fronts:

```bash
for spec in \
  "mnist cnn7 0.3 0_3 300 1000 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 cnn5 0.00784313725490196 2_255 300 1000 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 cnn7 0.00784313725490196 2_255 300 1000 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 wide_cnn7 0.00784313725490196 2_255 300 1000 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 narrow_cnn7 0.00784313725490196 2_255 300 1000 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 cnn9 0.00784313725490196 2_255 300 1000 mtl_ibp sabr shi crown_ibp_nofusion"; do
  set -- ${spec}
  dataset="$1"; network="$2"; eps="$3"; eps_tag="$4"; timeout="$5"; samples="$6"; shift 6
  for method in "$@"; do
    python papers/rethinking_evaluation_paradigms/mo_hpo/verify_front.py \
      --front "papers/rethinking_evaluation_paradigms/results/hpo/main/pareto_fronts/pareto_front_${method}_${network}_${dataset}_${eps}.csv" \
      --study-root papers/rethinking_evaluation_paradigms/results/hpo/main/optuna_studies \
      --dataset "${dataset}" \
      --network "${network}" \
      --method "${method}" \
      --eps "${eps}" \
      --timeout "${timeout}" \
      --test-samples "${samples}" \
      --results-root papers/rethinking_evaluation_paradigms/results/verification/main
  done
done
```

The utility loads checkpoints from the `source_study` and `config_hash`
columns in the front CSV and writes each `results.json` under
`papers/rethinking_evaluation_paradigms/results/verification/main/{dataset}/{network}/{eps}/{method}/{config_hash}`.
It verifies rows marked `subselected` by default; pass `--selection all` for
the full Pareto front.

For cluster-scale runs, both launchers read only final, subselected front
artifacts and share the same checkpoint-discovery function:

- `submitit_experiments/submit_complete_verification.py` submits one job per
  selected checkpoint and writes one `results.json`.
- `submitit_experiments/submit_chunked_complete_verification.py` splits each
  selected checkpoint by dataset index and writes one JSON per chunk to avoid
  concurrent writes. Auditing accumulates completed instances in an atomic
  `results.json`; `--instances-per-chunk` can then rechunk only missing indices.

Both default to a dry run; inspect the job list, then pass `--submit`. Use
`--audit-results` with the chunked launcher to report incomplete chunks and
refresh the authoritative result files. Stop old workers before auditing and
resubmitting with a different chunk size. They default to the main-paper fronts in
`results/hpo/main/pareto_fronts`; pass
`--fronts-root results/hpo/validation/pareto_fronts` together with
`--results-root results/verification/validation` for validation-split fronts.

## Paper Analysis Commands

The exact commands, archived-summary settings, and paper figure/table mapping
are maintained in the parent [`README.md`](../README.md).

For parallel-coordinate hyperparameter importance plots, use a Python 3.10
environment because `deepcave==1.3.4` requires it:

```bash
cd papers/rethinking_evaluation_paradigms/eval
python3.10 -m venv .venv-deepcave
source .venv-deepcave/bin/activate
pip install deepcave==1.3.4 optuna kaleido==0.2.1
python parallel_coordinates.py
```

## Notes

- The HPO runs are expensive. The paper used a SLURM cluster and one GPU per
  HPO job.
- `run_hpo.py` is scheduler-agnostic. Use `submitit`, SLURM arrays, GNU
  `parallel`, or your cluster launcher around the commands above.
- The publication fork also contains validation-split variants of some HPO
  experiments. Those use the same command structure, but the validation loader
  rather than the test loader is passed to `hpo`.
