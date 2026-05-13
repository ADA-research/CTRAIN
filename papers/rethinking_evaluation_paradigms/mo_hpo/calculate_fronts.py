import argparse
import csv
from pathlib import Path

import optuna


def dominates(left, right):
    return all(a >= b for a, b in zip(left, right)) and any(a > b for a, b in zip(left, right))


def pareto_trials(trials):
    completed = [
        trial for trial in trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.values is not None
    ]
    front = []
    for trial in completed:
        if not any(dominates(other.values, trial.values) for other in completed if other is not trial):
            front.append(trial)
    return sorted(front, key=lambda trial: (trial.values[0], trial.values[1]), reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Combine Optuna MO-HPO studies and export a Pareto front.")
    parser.add_argument("--study", action="append", required=True, help="Path to an optuna_study.db file.")
    parser.add_argument("--study-name", default="moctrain")
    parser.add_argument("--output", required=True, help="CSV file for the combined Pareto front.")
    parser.add_argument("--max-trials-per-study", type=int, default=None)
    args = parser.parse_args()

    all_trials = []
    for study_path in args.study:
        study = optuna.load_study(
            study_name=args.study_name,
            storage=f"sqlite:///{Path(study_path).resolve()}",
        )
        trials = study.trials
        if args.max_trials_per_study is not None:
            trials = trials[:args.max_trials_per_study]
        for trial in trials:
            trial.user_attrs["source_study"] = study_path
        all_trials.extend(trials)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    front = pareto_trials(all_trials)

    param_names = sorted({name for trial in front for name in trial.params})
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["nat_acc", "cert_acc", "adv_acc", "config_hash", "source_study", "study_trial", *param_names],
        )
        writer.writeheader()
        for trial in front:
            row = {
                "nat_acc": trial.values[0],
                "cert_acc": trial.values[1],
                "adv_acc": trial.user_attrs.get("adv_acc"),
                "config_hash": trial.user_attrs.get("config_hash"),
                "source_study": trial.user_attrs.get("source_study"),
                "study_trial": trial.number,
            }
            row.update(trial.params)
            writer.writerow(row)

    print(f"Wrote {len(front)} Pareto-optimal trials to {output_path}")


if __name__ == "__main__":
    main()
