<p align="center">
  <img src="../../docs/assets/mo-hpo-motivation.png" alt="Motivating Pareto-front comparison for MO-HPO" width="75%">
</p>

<h1 align="center">Rethinking Evaluation Paradigms in IBP-based Certified Training</h1>

<!-- <p align="center">
  <strong>From single reported configurations to fair Pareto-front comparisons.</strong>
</p> -->

<!-- Image wildcard: replace ../../docs/assets/mo-hpo-motivation.svg with a motivating plot from eval/plots or a final paper figure. -->

This directory contains the code and artifacts for the paper preprint
**"Rethinking Evaluation Paradigms in IBP-based Certified Training"**.

The paper argues that IBP-based certified training methods should not be
compared by a single hand-tuned configuration. Natural accuracy and certified
accuracy form a trade-off, so the meaningful object of comparison is a Pareto
front. The experiments in this directory implement that protocol: discover
strong Pareto candidates with multi-objective HPO, verify the selected
checkpoints with complete verification, and analyse the resulting fronts.

This folder is a paper artifact area. It is intentionally separate from the
installable `CTRAIN` Python package and is excluded from source distributions.

## 🎯 Paper Summary

The paper studies certified training methods whose robustness certificates are
based on IBP-style bounds, including IBP
([Gowal et al., 2018](https://arxiv.org/abs/1810.12715);
[Shi et al., 2021](https://arxiv.org/abs/2102.06700)), CROWN-IBP
([Zhang et al., 2020](https://arxiv.org/abs/1906.06316)), SABR
([Müller et al., 2023](https://arxiv.org/pdf/2210.04871)), and MTL-IBP
([De Palma et al., 2023](https://arxiv.org/pdf/2305.13991)). These methods
expose many interacting hyperparameters: ramp schedules, loss weights,
regularisation terms, method specific controls, optimiser settings, and
architecture choices. Reporting one configuration per method can therefore be
misleading because a method may look weak simply because its natural-certified
trade-off was undertuned.

The proposed evaluation replaces single-point comparisons with:

1. Multi-objective HPO over natural validation accuracy and certified
   validation accuracy.
2. Three independent HPO seeds per benchmark.
3. Combination of seed-wise fronts into a single method-level Pareto front.
4. Subselection of non-redundant candidates before expensive complete
   verification.
5. Final reporting using complete verification rather than the incomplete
   verification objective used during HPO.

The resulting fronts show that automated tuning often finds configurations that
dominate previously reported results, that older methods can still contribute
to the combined state-of-the-art front, and that different methods are
competitive in different regions of the trade-off.

## 🗂️ Directory Layout

```text
papers/rethinking_evaluation_paradigms/
|-- README.md
|-- ORIGINAL_README.md
|-- requirements-paper.txt
|-- mo_hpo/
|   |-- README.md
|   |-- run_hpo.py
|   |-- calculate_fronts.py
|   `-- verify_front.py
|-- submitit_experiments/
|   |-- submit_hpo.py
|   |-- submit_hpo_validation_split.py
|   |-- calculate_fronts.py
|   |-- submit_complete_verification.py
|   `-- submit_chunked_complete_verification.py
|-- eval/
|   |-- combine_results.py
|   |-- eval_nat_acc.py
|   |-- plot.py
|   |-- tables.py
|   |-- front_analysis.py
|   |-- front_status.py
|   |-- parallel_coordinates.py
|   |-- plot_motivation.py
|   |-- verification_times.py
|   `-- verify_correlation.py
`-- results/
    |-- clean_classification/
    `-- verification/
```

The most important maintained entrypoints are in `mo_hpo/`. The
`submitit_experiments/` directory preserves the original cluster-scale SLURM
scripts. The `eval/` directory contains the analysis code used for plots,
tables, convergence summaries, verification-time analysis, and hyperparameter
importance visualisations.

## ⚙️ Setup

Run from the repository root:

```bash
cd /storage/work/kaulen/ctrain_dev/CTRAIN
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The paper runs use Optuna with the BoTorch sampler. The package metadata
includes the relevant HPO dependencies. The original frozen environment is
recorded in `requirements-paper.txt`.

For the maintained `mo_hpo/` scripts, pass paths explicitly with
`--data-root`, `--output-root`, and `--results-root`. The preserved
`submitit_experiments/` scripts also support these environment overrides:

```bash
export CTRAIN_DATA_ROOT=/path/to/data
export CTRAIN_PAPER_RESULTS_ROOT=/path/to/paper/results
```

If unset, the submitit scripts use their repository-relative defaults.

## 🔁 End-to-End Workflow

### 1. Run Multi-Objective HPO 🚀

Use `mo_hpo/run_hpo.py` to launch one HPO job. It builds the dataset loader,
network, certified-training wrapper, and Optuna study, then stores the study
database and all trained checkpoints.

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

Important arguments:

- `--dataset`: `cifar10`, `mnist`, `gtsrb`, or `tinyimagenet`.
- `--network`: `cnn3`, `cnn5`, `cnn7`, `cnn9`, `cnn11`, `wide_cnn7`,
  `narrow_cnn7`, or `resnet18`.
- `--method`: `mtl_ibp`, `sabr`, `shi`, `crown_ibp`, or
  `crown_ibp_nofusion`.
- `--eps`: perturbation radius, for example `0.00784313725490196` for `2/255`.
- `--sampler`: `botorch` for paper-style runs or `nsgaii` for lighter local
  smoke tests.
- `--complete-verify`: optionally use complete verification inside the HPO
  objective; the paper workflow instead uses incomplete verification for HPO
  and complete verification only after Pareto filtering.

Each run writes:

```text
results/mo_hpo/{dataset}_{network}_{method}_{eps}_{seed}/optuna_study.db
results/mo_hpo/{dataset}_{network}_{method}_{eps}_{seed}/nets/{config_hash}.pt
```

The paper uses 100 trials per seed and three seeds per benchmark.

### 2. Combine Seed-Wise Pareto Fronts 📈

After all three seeds finish for a benchmark/method combination, combine the
Optuna studies:

```bash
python papers/rethinking_evaluation_paradigms/mo_hpo/calculate_fronts.py \
  --study papers/rethinking_evaluation_paradigms/results/mo_hpo/cifar10_cnn7_mtl_ibp_0.00784313725490196_0/optuna_study.db \
  --study papers/rethinking_evaluation_paradigms/results/mo_hpo/cifar10_cnn7_mtl_ibp_0.00784313725490196_1/optuna_study.db \
  --study papers/rethinking_evaluation_paradigms/results/mo_hpo/cifar10_cnn7_mtl_ibp_0.00784313725490196_2/optuna_study.db \
  --output papers/rethinking_evaluation_paradigms/results/mo_hpo/fronts/cifar10_cnn7_mtl_ibp_2_255_front.csv
```

The output CSV contains the incomplete-verification objectives, Optuna trial
number, source study path, checkpoint `config_hash`, and all hyperparameters.
The `source_study` and `config_hash` columns are used later to locate each
checkpoint.

### 3. Run Complete Verification ✅

Complete verification is applied only to candidates on the combined front:

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

The paper uses:

- `1000s` complete-verification timeouts for the main CNN7 CIFAR-10 and Tiny
  ImageNet experiments.
- `300s` complete-verification timeouts for MNIST and architecture-appendix
  analyses.
- The first `1000` test samples for the reduced-cost architecture appendix.

Cluster-scale verification can be launched through
`submitit_experiments/submit_chunked_complete_verification.py`. That script
defaults to dry-run mode; inspect the generated job list before setting
`DRY_RUN=False`.

### 4. Evaluate Clean Accuracy 🧪

Clean classification results are stored under `results/clean_classification`.
The original paper workflow uses:

```bash
cd papers/rethinking_evaluation_paradigms/eval
python eval_nat_acc.py
```

### 5. Combine Results and Generate Paper Artifacts 📊

The plotting and table scripts use paths relative to `eval/`, so run them from
that directory:

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

These scripts produce:

- Combined natural/certified result summaries.
- Pareto-front plots for individual methods and method comparisons.
- Tables for main and appendix results.
- Hypervolume/convergence summaries over HPO progress.
- Motivation plots comparing tuned fronts with literature results.
- Verification-time analyses for different timeout budgets.
- Correlation analysis between incomplete and complete verification outcomes.

For parallel-coordinate hyperparameter-importance plots, use Python 3.10
because `deepcave==1.3.4` requires it:

```bash
cd papers/rethinking_evaluation_paradigms/eval
python3.10 -m venv .venv-deepcave
source .venv-deepcave/bin/activate
pip install deepcave==1.3.4 optuna kaleido==0.2.1
python parallel_coordinates.py
```

## 📌 Main Paper Benchmarks

The main CNN7 experiments use:

| Dataset | Radius | Epochs | Methods |
| --- | ---: | ---: | --- |
| CIFAR-10 | `2/255` | 160 | `mtl_ibp`, `sabr`, `shi`, `crown_ibp_nofusion` |
| CIFAR-10 | `8/255` | 260 | `mtl_ibp`, `sabr`, `shi`, `crown_ibp_nofusion` |
| Tiny ImageNet | `1/255` | 160 | `mtl_ibp`, `sabr`, `shi`, `crown_ibp` |

The appendix additionally includes MNIST at `eps=0.3` and architecture studies
on CIFAR-10, including CNN5, CNN7, CNN7 Wide, CNN7 Narrow, and CNN9. See
`mo_hpo/README.md` for batch commands covering the main paper and appendix
grids.

## 🧰 Script Responsibilities

- `mo_hpo/run_hpo.py`: scheduler-agnostic entrypoint for one HPO job.
- `mo_hpo/calculate_fronts.py`: loads multiple Optuna studies and exports the
  nondominated trials as a combined Pareto-front CSV.
- `mo_hpo/verify_front.py`: loads checkpoints listed in a front CSV and runs
  complete verification.
- `submitit_experiments/submit_hpo.py`: original SLURM submission script for
  HPO grids.
- `submitit_experiments/submit_hpo_validation_split.py`: validation-split HPO
  variant from the publication code.
- `submitit_experiments/submit_complete_verification.py`: original complete
  verification submission script.
- `submitit_experiments/submit_chunked_complete_verification.py`: chunked
  complete verification launcher designed to avoid concurrent result writes.
- `eval/combine_results.py`: merges verification and clean-accuracy results.
- `eval/plot.py` and `eval/tables.py`: produce paper plots and tables.
- `eval/front_analysis.py`: reports which methods or architectures contribute
  to combined fronts.
- `eval/front_status.py`: analyses front development and hypervolume progress.
- `eval/parallel_coordinates.py`: creates DeepCAVE-based hyperparameter
  importance plots.
- `eval/verification_times.py`: analyses verification runtime and timeout
  sensitivity.
- `eval/verify_correlation.py`: compares incomplete-verification objectives
  against complete-verification outcomes.

## ⚠️ Reproducibility Notes

- Full reproduction is computationally expensive. The paper experiments were
  run on SLURM clusters with GPU jobs.
- Checkpoints from the original paper runs are large and are not all included
  in this repository.
- Existing `results/` artifacts document the paper run outputs available in
  this checkout.

## 📝 Citation

If you use this code, the HPO workflow, or the included artifacts, please cite:

```bibtex
@inproceedings{KauEtAl26,
  title = {Rethinking Evaluation Paradigms in IBP-based Certified Training},
  author = {Kaulen, Konstantin and Shavit, Hadar and Hoos, Holger H},
  booktitle={To appear in: Proceedings of the 43rd International Conference on Machine Learning (ICML 2026)},
  year="2026"
}
```
