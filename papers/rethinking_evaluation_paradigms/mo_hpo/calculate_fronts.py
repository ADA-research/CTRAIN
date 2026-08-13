"""Combine seed-wise Optuna studies into one portable Pareto-front CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import optuna

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from front_utils import (  # noqa: E402
    pareto_trials,
    subselected_trials,
    trial_config_hash,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine Optuna MO-HPO studies and export one Pareto-front CSV."
    )
    parser.add_argument(
        "--study",
        action="append",
        required=True,
        help="Path to an optuna_study.db file; repeat once per seed.",
    )
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--method", required=True)
    parser.add_argument("--eps", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-trials-per-study", type=int, default=100)
    parser.add_argument("--subselection-distance", type=float, default=0.05)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow fewer completed trials than the requested per-study budget.",
    )
    return parser.parse_args()


def load_study(path: Path, study_name: str | None):
    storage = f"sqlite:///{path.resolve()}"
    if study_name is None:
        summaries = optuna.get_all_study_summaries(storage=storage)
        if len(summaries) != 1:
            raise ValueError(
                f"Expected exactly one study in {path}, found {len(summaries)}; "
                "pass --study-name explicitly."
            )
        study_name = summaries[0].study_name
    return optuna.load_study(study_name=study_name, storage=storage)


def main():
    args = parse_args()
    if args.max_trials_per_study <= 0:
        raise ValueError("--max-trials-per-study must be positive")

    trials = []
    sources = {}
    for supplied_path in args.study:
        path = Path(supplied_path)
        study_trials = load_study(path, args.study_name).trials[
            : args.max_trials_per_study
        ]
        completed = sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            and trial.values is not None
            for trial in study_trials
        )
        if completed != args.max_trials_per_study and not args.allow_incomplete:
            raise ValueError(
                f"{path} has {completed} completed trials among the first "
                f"{args.max_trials_per_study}; pass --allow-incomplete to continue."
            )
        for trial in study_trials:
            sources[id(trial)] = path.parent.name
        trials.extend(study_trials)

    front = pareto_trials(trials)
    selected_ids = {
        id(trial)
        for trial in subselected_trials(front, args.subselection_distance)
    }
    param_names = sorted({name for trial in front for name in trial.params})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "nat_acc",
                "cert_acc",
                "adv_acc",
                "config_hash",
                "subselected",
                "source_study",
                "study_trial",
                *param_names,
            ],
        )
        writer.writeheader()
        for trial in front:
            row = {
                "nat_acc": trial.values[0],
                "cert_acc": trial.values[1],
                "adv_acc": trial.user_attrs.get("adv_acc"),
                "config_hash": trial_config_hash(trial, args.method, args.eps),
                "subselected": str(id(trial) in selected_ids).lower(),
                "source_study": sources[id(trial)],
                "study_trial": trial.number,
            }
            row.update(trial.params)
            writer.writerow(row)

    print(
        f"Wrote {len(front)} Pareto points "
        f"({len(selected_ids)} subselected) to {args.output}"
    )


if __name__ == "__main__":
    main()
