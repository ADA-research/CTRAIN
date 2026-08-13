"""Shared rules for constructing and identifying MO-HPO Pareto fronts."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import numpy as np
import optuna
from scipy.cluster.hierarchy import fclusterdata

from CTRAIN.model_wrappers.configs import (
    build_crown_ibp_config_space,
    build_mtl_ibp_config_space,
    build_sabr_config_space,
    build_shi_config_space,
)


def trial_is_feasible(trial: optuna.trial.FrozenTrial) -> bool:
    constraints = trial.system_attrs.get(
        "constraints",
        trial.user_attrs.get("constraints", trial.user_attrs.get("constraint")),
    )
    return constraints is None or all(float(value) <= 0 for value in constraints)


def completed_feasible_trials(
    trials: Iterable[optuna.trial.FrozenTrial],
) -> list[optuna.trial.FrozenTrial]:
    return [
        trial
        for trial in trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.values is not None
        and trial_is_feasible(trial)
    ]


def dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a >= b for a, b in zip(left, right)) and any(
        a > b for a, b in zip(left, right)
    )


def pareto_trials(
    trials: Iterable[optuna.trial.FrozenTrial],
) -> list[optuna.trial.FrozenTrial]:
    feasible = completed_feasible_trials(trials)
    front = [
        trial
        for trial in feasible
        if not any(
            other is not trial and dominates(other.values, trial.values)
            for other in feasible
        )
    ]
    return sorted(
        front,
        key=lambda trial: (trial.values[0], trial.values[1], -trial.number),
        reverse=True,
    )


def config_hash_epochs(eps: float) -> int:
    if abs(eps - 8 / 255) < 1e-12:
        return 260
    if abs(eps - 0.3) < 1e-12:
        return 70
    return 160


def config_space_for_hash(method: str, eps: float):
    epochs = config_hash_epochs(eps)
    if method in {"crown_ibp", "crown_ibp_nofusion"}:
        return build_crown_ibp_config_space(epochs=epochs, eps=eps)
    if method == "sabr":
        return build_sabr_config_space(epochs=epochs, eps=eps)
    if method == "mtl_ibp":
        return build_mtl_ibp_config_space(epochs=epochs, eps=eps)
    if method == "shi":
        return build_shi_config_space(epochs=epochs, eps=eps)
    raise ValueError(f"Cannot reconstruct a configuration hash for {method!r}")


def trial_config_hash(
    trial: optuna.trial.FrozenTrial, method: str, eps: float
) -> str:
    stored_hash = trial.user_attrs.get("config_hash")
    if stored_hash:
        return str(stored_hash)

    config = {}
    config_space = config_space_for_hash(method, eps)
    for name in config_space:
        hyperparameter = config_space[name]
        if hyperparameter.__class__.__name__ == "Constant":
            config[name] = getattr(
                hyperparameter, "value", hyperparameter.default_value
            )
        else:
            try:
                config[name] = trial.params[name]
            except KeyError as exc:
                raise ValueError(
                    f"Trial {trial.number} has no value for hash parameter {name!r}"
                ) from exc
    return configuration_hash(config)


def configuration_hash(config: object, chars: int = 32) -> str:
    """Match SMAC's configuration hash without importing its full stack."""
    return hashlib.sha1(str(config).encode("utf-8")).hexdigest()[:chars]


def subselected_trials(
    trials: Sequence[optuna.trial.FrozenTrial], distance: float = 0.05
) -> list[optuna.trial.FrozenTrial]:
    if distance <= 0:
        raise ValueError("Subselection distance must be positive")
    if len(trials) <= 5:
        return list(trials)

    points = np.asarray([trial.values for trial in trials], dtype=float)
    normalized = points.copy()
    for column in range(normalized.shape[1]):
        span = np.ptp(normalized[:, column])
        normalized[:, column] = (
            0 if span == 0 else (normalized[:, column] - normalized[:, column].min()) / span
        )

    labels = fclusterdata(normalized, distance, criterion="distance")
    selected = []
    for label in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        # Publication rule: retain the largest-natural-accuracy point per cluster.
        best = min(
            indices,
            key=lambda index: (
                -points[index, 0],
                -points[index, 1],
                trials[index].number,
            ),
        )
        selected.append(trials[best])
    return selected
