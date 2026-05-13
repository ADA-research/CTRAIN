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

The HPO objective uses natural validation accuracy and certified validation
accuracy from incomplete verification. The final reported certified accuracy
should be computed with complete verification.

## Setup

Run from the repository root:

```bash
cd /storage/work/kaulen/ctrain_dev/CTRAIN
python -m venv .venv
source .venv/bin/activate
pip install -e .
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
  --output-root papers/rethinking_evaluation_paradigms/results/mo_hpo
```

Each run writes:

- Optuna study: `papers/rethinking_evaluation_paradigms/results/mo_hpo/{dataset}_{network}_{method}_{eps}_{seed}/optuna_study.db`
- Checkpoints: `papers/rethinking_evaluation_paradigms/results/mo_hpo/{dataset}_{network}_{method}_{eps}_{seed}/nets/{config_hash}.pt`

Use `--sampler nsgaii` for dependency-light tests. Use the default
`--sampler botorch` for paper-style runs.

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
      --output-root papers/rethinking_evaluation_paradigms/results/mo_hpo
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
      --output-root papers/rethinking_evaluation_paradigms/results/mo_hpo
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
      --output-root papers/rethinking_evaluation_paradigms/results/mo_hpo
  done
done
```

## Appendix HPO Runs

MNIST, `eps=0.3`, CNN7, 70 epochs, 300s complete-verification cutoff in the
paper analysis:

```bash
for method in mtl_ibp sabr shi crown_ibp; do
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
      --output-root papers/rethinking_evaluation_paradigms/results/mo_hpo
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
        --output-root papers/rethinking_evaluation_paradigms/results/mo_hpo
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
        --output-root papers/rethinking_evaluation_paradigms/results/mo_hpo
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
        --output-root papers/rethinking_evaluation_paradigms/results/mo_hpo
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
      --output-root papers/rethinking_evaluation_paradigms/results/mo_hpo_val
  done
done
```

## Combine Seed-Wise Fronts

After the three HPO seeds finish, combine each group of studies. Example for
CIFAR-10, CNN7, MTL-IBP, `eps=2/255`:

```bash
python papers/rethinking_evaluation_paradigms/mo_hpo/calculate_fronts.py \
  --study papers/rethinking_evaluation_paradigms/results/mo_hpo/cifar10_cnn7_mtl_ibp_0.00784313725490196_0/optuna_study.db \
  --study papers/rethinking_evaluation_paradigms/results/mo_hpo/cifar10_cnn7_mtl_ibp_0.00784313725490196_1/optuna_study.db \
  --study papers/rethinking_evaluation_paradigms/results/mo_hpo/cifar10_cnn7_mtl_ibp_0.00784313725490196_2/optuna_study.db \
  --output papers/rethinking_evaluation_paradigms/results/mo_hpo/fronts/cifar10_cnn7_mtl_ibp_2_255_front.csv
```

Batch command for the main-paper fronts:

```bash
mkdir -p papers/rethinking_evaluation_paradigms/results/mo_hpo/fronts

for spec in \
  "cifar10 cnn7 0.00784313725490196 2_255 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 cnn7 0.03137254901960784 8_255 mtl_ibp sabr shi crown_ibp_nofusion" \
  "tinyimagenet cnn7 0.00392156862745098 1_255 mtl_ibp sabr shi crown_ibp"; do
  set -- ${spec}
  dataset="$1"; network="$2"; eps="$3"; eps_tag="$4"; shift 4
  for method in "$@"; do
    python papers/rethinking_evaluation_paradigms/mo_hpo/calculate_fronts.py \
      --study "papers/rethinking_evaluation_paradigms/results/mo_hpo/${dataset}_${network}_${method}_${eps}_0/optuna_study.db" \
      --study "papers/rethinking_evaluation_paradigms/results/mo_hpo/${dataset}_${network}_${method}_${eps}_1/optuna_study.db" \
      --study "papers/rethinking_evaluation_paradigms/results/mo_hpo/${dataset}_${network}_${method}_${eps}_2/optuna_study.db" \
      --output "papers/rethinking_evaluation_paradigms/results/mo_hpo/fronts/${dataset}_${network}_${method}_${eps_tag}_front.csv"
  done
done
```

Batch command for MNIST and the architecture appendix:

```bash
for spec in \
  "mnist cnn7 0.3 0_3 mtl_ibp sabr shi crown_ibp" \
  "cifar10 cnn5 0.00784313725490196 2_255 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 cnn7 0.00784313725490196 2_255 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 wide_cnn7 0.00784313725490196 2_255 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 narrow_cnn7 0.00784313725490196 2_255 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 cnn9 0.00784313725490196 2_255 mtl_ibp sabr shi crown_ibp_nofusion"; do
  set -- ${spec}
  dataset="$1"; network="$2"; eps="$3"; eps_tag="$4"; shift 4
  for method in "$@"; do
    python papers/rethinking_evaluation_paradigms/mo_hpo/calculate_fronts.py \
      --study "papers/rethinking_evaluation_paradigms/results/mo_hpo/${dataset}_${network}_${method}_${eps}_0/optuna_study.db" \
      --study "papers/rethinking_evaluation_paradigms/results/mo_hpo/${dataset}_${network}_${method}_${eps}_1/optuna_study.db" \
      --study "papers/rethinking_evaluation_paradigms/results/mo_hpo/${dataset}_${network}_${method}_${eps}_2/optuna_study.db" \
      --output "papers/rethinking_evaluation_paradigms/results/mo_hpo/fronts/${dataset}_${network}_${method}_${eps_tag}_front.csv"
  done
done
```

The CSV contains the incomplete-verification objectives, the Optuna trial
number, the `config_hash`, and all hyperparameters needed to locate the
checkpoint in the corresponding `nets/` directory.

## Complete Verification

The paper uses complete verification only after Pareto filtering/subselection:

- Main CNN7 CIFAR-10 and Tiny ImageNet: alpha-beta-CROWN timeout `1000s`.
- MNIST and architecture appendix: timeout `300s`.
- Architecture appendix: first `1000` test samples in the reported reduced-cost
  analysis.

Verify one combined front serially:

```bash
python papers/rethinking_evaluation_paradigms/mo_hpo/verify_front.py \
  --front papers/rethinking_evaluation_paradigms/results/mo_hpo/fronts/cifar10_cnn7_mtl_ibp_2_255_front.csv \
  --dataset cifar10 \
  --network cnn7 \
  --method mtl_ibp \
  --eps 0.00784313725490196 \
  --timeout 1000 \
  --test-samples 10000 \
  --abcrown-batch-size 1024 \
  --results-root papers/rethinking_evaluation_paradigms/results/verification
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
      --front "papers/rethinking_evaluation_paradigms/results/mo_hpo/fronts/${dataset}_${network}_${method}_${eps_tag}_front.csv" \
      --dataset "${dataset}" \
      --network "${network}" \
      --method "${method}" \
      --eps "${eps}" \
      --timeout "${timeout}" \
      --test-samples "${samples}" \
      --results-root papers/rethinking_evaluation_paradigms/results/verification
  done
done
```

Batch command for MNIST and architecture-appendix fronts:

```bash
for spec in \
  "mnist cnn7 0.3 0_3 300 10000 mtl_ibp sabr shi crown_ibp" \
  "cifar10 cnn5 0.00784313725490196 2_255 300 1000 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 cnn7 0.00784313725490196 2_255 300 1000 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 wide_cnn7 0.00784313725490196 2_255 300 1000 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 narrow_cnn7 0.00784313725490196 2_255 300 1000 mtl_ibp sabr shi crown_ibp_nofusion" \
  "cifar10 cnn9 0.00784313725490196 2_255 300 1000 mtl_ibp sabr shi crown_ibp_nofusion"; do
  set -- ${spec}
  dataset="$1"; network="$2"; eps="$3"; eps_tag="$4"; timeout="$5"; samples="$6"; shift 6
  for method in "$@"; do
    python papers/rethinking_evaluation_paradigms/mo_hpo/verify_front.py \
      --front "papers/rethinking_evaluation_paradigms/results/mo_hpo/fronts/${dataset}_${network}_${method}_${eps_tag}_front.csv" \
      --dataset "${dataset}" \
      --network "${network}" \
      --method "${method}" \
      --eps "${eps}" \
      --timeout "${timeout}" \
      --test-samples "${samples}" \
      --results-root papers/rethinking_evaluation_paradigms/results/verification
  done
done
```

The utility loads checkpoints from the `source_study` and `config_hash`
columns in the front CSV and writes each `results.json` under
`papers/rethinking_evaluation_paradigms/results/verification/{dataset}/{network}/{eps}/{method}/{config_hash}`.
For full-scale runs, launch each `verify_front.py` invocation through SLURM or
your cluster scheduler.

## Paper Analysis Commands

The original publication repository contains the following analysis scripts.
They use paths relative to `papers/rethinking_evaluation_paradigms/eval`, so
run them from that directory. After placing verification results under
`results/verification` and clean classification results under
`results/clean_classification`, run:

```bash
cd papers/rethinking_evaluation_paradigms/eval
python combine_results.py
python plot.py
python tables.py
python front_analysis.py
python front_status.py
python plot_motivation.py
python verification_times.py
python verify_correlation.py
```

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
