"""Small, dependency-light readers for archived Optuna result directories."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StudySpec:
    dataset: str
    architecture: str
    method: str
    eps: str
    seed: int
    complete: bool | None
    path: Path
    database: Path | None = None

    @property
    def group_key(self):
        return (self.dataset, self.architecture, self.method, self.eps)

    @property
    def database_path(self) -> Path:
        return self.database or self.path / "optuna_study.db"


def parse_run_dir(
    path: Path,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    database: Path | None = None,
) -> StudySpec | None:
    tokens = path.name.split("_")
    complete = None
    if len(tokens) >= 2 and tokens[-2] == "complete":
        complete = tokens[-1] == "True"
        tokens = tokens[:-2]

    dataset = next(
        (name for name in datasets if path.name.startswith(f"{name}_")), None
    )
    if dataset is None or len(tokens) < 5:
        return None
    try:
        seed = int(tokens[-1])
    except ValueError:
        return None
    eps = tokens[-2]
    middle = tokens[len(dataset.split("_")) : -2]
    for method in sorted(methods, key=len, reverse=True):
        method_tokens = method.split("_")
        if middle[-len(method_tokens) :] != method_tokens:
            continue
        architecture = "_".join(middle[: -len(method_tokens)])
        if architecture:
            return StudySpec(
                dataset, architecture, method, eps, seed, complete, path, database
            )
    return None


def find_studies(
    root: Path,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    *,
    selected_datasets: set[str] | None = None,
    selected_methods: set[str] | None = None,
    complete: bool | None = None,
) -> list[StudySpec]:
    specs = []
    if not root.exists():
        return specs
    databases = [(database.parent, database) for database in root.glob("*/optuna_study.db")]
    suffix = "_optuna_study.db"
    databases.extend(
        (database.with_name(database.name[: -len(suffix)]), database)
        for database in root.glob(f"*{suffix}")
    )
    for run_path, database in databases:
        spec = parse_run_dir(run_path, datasets, methods, database)
        if spec is None:
            continue
        if selected_datasets is not None and spec.dataset not in selected_datasets:
            continue
        if selected_methods is not None and spec.method not in selected_methods:
            continue
        if complete is not None and spec.complete is not None and spec.complete != complete:
            continue
        specs.append(spec)
    return sorted(specs, key=lambda spec: (spec.group_key, spec.seed))


def load_trial_points(spec: StudySpec, max_trials: int | None):
    connection = sqlite3.connect(spec.database_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT trials.trial_id, trials.number,
                   trial_values.objective, trial_values.value
            FROM trials
            JOIN trial_values ON trial_values.trial_id = trials.trial_id
            WHERE trials.state = 'COMPLETE'
            ORDER BY trials.number, trial_values.objective
            """
        ).fetchall()
        constraint_rows = connection.execute(
            """
            SELECT trial_id, value_json
            FROM trial_user_attributes
            WHERE key IN ('constraint', 'constraints')
            """
        ).fetchall()
    finally:
        connection.close()

    by_trial = {}
    for row in rows:
        trial = by_trial.setdefault(
            row["trial_id"], {"trial_number": int(row["number"])}
        )
        trial[int(row["objective"])] = float(row["value"])
    for row in constraint_rows:
        by_trial.setdefault(row["trial_id"], {})["constraints"] = row["value_json"]

    points = []
    for trial in by_trial.values():
        number = trial.get("trial_number")
        if number is None or 0 not in trial or 1 not in trial:
            continue
        if max_trials is not None and number >= max_trials:
            continue
        encoded_constraints = trial.get("constraints")
        constraints = json.loads(encoded_constraints) if encoded_constraints else []
        if not isinstance(constraints, list):
            constraints = [constraints]
        points.append(
            {
                "trial_number": number,
                "clean_acc": trial[0],
                "certified_acc": trial[1],
                "feasible": all(float(value) <= 0 for value in constraints),
                "constraints": encoded_constraints,
            }
        )
    return points
