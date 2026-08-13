"""Reproduce Table 4 by matching incomplete and complete results by hash."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr


PAPER_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != SCRIPT_ROOT
]
sys.path.insert(0, str(PAPER_ROOT.parents[1]))
sys.path.insert(0, str(PAPER_ROOT / "mo_hpo"))

from front_utils import pareto_trials, trial_config_hash  # noqa: E402


def incomplete_front(dataset, network, method, eps, trial_count=100):
    trials = []
    for seed in range(3):
        database = (
            PAPER_ROOT
            / "results"
            / "hpo"
            / "main"
            / "optuna_studies"
            / f"{dataset}_{network}_{method}_{eps}_{seed}_optuna_study.db"
        )
        study = optuna.load_study(
            study_name="moctrain", storage=f"sqlite:///{database}"
        )
        trials.extend(study.trials[:trial_count])

    result = {}
    for trial in pareto_trials(trials):
        config_hash = trial_config_hash(trial, method, eps)
        if config_hash in result:
            raise ValueError(f"Duplicate incomplete configuration hash {config_hash}")
        result[config_hash] = float(trial.values[1]) * 100
    return result


def complete_results(method, dataset, network, eps):
    with (PAPER_ROOT / "results/verification/main/summary_results.json").open() as handle:
        summary = json.load(handle)

    result = {}
    for row in summary:
        if not (
            row["dataset"] == dataset
            and row["architecture"] == network
            and abs(float(row["eps"]) - float(eps)) < 1e-12
            and row["cert_train_method"] == method
        ):
            continue
        config_hash = row["hash"]
        if config_hash in result:
            raise ValueError(f"Duplicate complete configuration hash {config_hash}")
        result[config_hash] = float(row["certified_accuracy"])
    return result


def compare_incomplete_vs_complete():
    rows = []
    methods_by_dataset = {
        "cifar10": ["mtl_ibp", "sabr", "shi", "crown_ibp_nofusion"],
        "tinyimagenet": ["mtl_ibp", "sabr", "shi", "crown_ibp"],
    }
    for dataset, methods in methods_by_dataset.items():
        eps_values = [2 / 255, 8 / 255] if dataset == "cifar10" else [1 / 255]
        for method in methods:
            for eps in eps_values:
                incomplete = incomplete_front(dataset, "cnn7", method, eps)
                complete = complete_results(method, dataset, "cnn7", eps)
                missing = sorted(set(complete) - set(incomplete))
                if missing:
                    raise ValueError(
                        f"{dataset}/cnn7/{method}/{eps}: complete configurations "
                        f"are absent from the incomplete Pareto front: {missing}"
                    )
                hashes = sorted(set(incomplete) & set(complete))
                if len(hashes) < 2:
                    raise ValueError(
                        f"{dataset}/cnn7/{method}/{eps}: only {len(hashes)} matches"
                    )
                incomplete_cert = [incomplete[key] for key in hashes]
                complete_cert = [complete[key] for key in hashes]
                rho, p_value = spearmanr(incomplete_cert, complete_cert)
                rows.append(
                    {
                        "Dataset": dataset,
                        "Network": "cnn7",
                        "Method": method,
                        "Epsilon": eps,
                        "Incomplete_Points": len(incomplete),
                        "Complete_Points": len(complete),
                        "Correlation_Points": len(hashes),
                        "Spearman_Rho": rho,
                        "P_Value": p_value,
                    }
                )
                print(
                    f"{dataset}_cnn7_{method}_{eps}: "
                    f"ρ={rho:.4f}, p={p_value:.4f}, n={len(hashes)}"
                )
    return rows


def epsilon_label(eps):
    for value, label in ((2 / 255, "2/255"), (8 / 255, "8/255"), (1 / 255, "1/255")):
        if abs(eps - value) < 1e-12:
            return label
    return f"{eps:.4f}"


def generate_correlation_table(rows):
    method_order = {
        "shi": 0,
        "crown_ibp_nofusion": 1,
        "crown_ibp": 1,
        "sabr": 2,
        "mtl_ibp": 3,
    }
    rows = sorted(
        rows,
        key=lambda row: (
            row["Dataset"] != "cifar10",
            row["Epsilon"],
            method_order[row["Method"]],
        ),
    )
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Spearman Rank Correlation: Incomplete vs Complete Verification}",
        "\\label{tab:verification_correlation}",
        "\\begin{tabular}{llll|rrr}",
        "\\toprule",
        "Dataset & Network & Method & $\\epsilon$ & $\\rho$ & $p$-value & $n$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        dataset = "CIFAR-10" if row["Dataset"] == "cifar10" else "Tiny-ImageNet"
        rho = 0.0 if np.isnan(row["Spearman_Rho"]) else row["Spearman_Rho"]
        p_value = 1.0 if np.isnan(row["P_Value"]) else row["P_Value"]
        lines.append(
            f"{dataset} & CNN7 & {row['Method'].upper()} & "
            f"{epsilon_label(row['Epsilon'])} & {rho:.4f} & {p_value:.4e} & "
            f"{row['Correlation_Points']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    output_dir = PAPER_ROOT / "tables" / "main"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "verification_correlation.tex").write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(
        output_dir / "verification_correlation.csv", index=False
    )


if __name__ == "__main__":
    print("Analyzing correlation between incomplete and complete verification...")
    correlation_results = compare_incomplete_vs_complete()
    rhos = np.asarray([row["Spearman_Rho"] for row in correlation_results])
    print(f"Average Spearman rho (n={np.sum(~np.isnan(rhos))}): {np.nanmean(rhos):.4f}")
    generate_correlation_table(correlation_results)
