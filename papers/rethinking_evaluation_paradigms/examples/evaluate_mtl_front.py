"""Select and evaluate a published MTL-IBP Pareto-front checkpoint.

Run this file from the CTRAIN repository root. By default it prints the
CIFAR-10/CNN7/MTL-IBP front at epsilon=2/255, selects the first row (highest
reported complete certified accuracy), and evaluates it on all 10,000 CIFAR-10
test examples with CTRAIN's standard clean/IBP/PGD evaluation pipeline.
CUDA is strongly recommended because CTRAIN's default PGD evaluation uses 30
restarts of 100 steps each.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from papers.rethinking_evaluation_paradigms.model_hub import (  # noqa: E402
    list_models,
    load_model,
)


def parse_epsilon(value: str) -> float:
    """Accept decimal radii as well as convenient values such as ``2/255``."""
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", maxsplit=1)
    return float(numerator) / float(denominator)


def complete_evaluation(model: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluations = model["evaluations"]
    if len(evaluations) != 1:
        raise ValueError(
            f"Expected one canonical evaluation for {model['model_id']}, "
            f"found {len(evaluations)}"
        )
    return evaluations[0]


def print_front(front: list[dict[str, Any]]) -> None:
    print("\nPublished complete-verification MTL-IBP Pareto front")
    print(
        f"{'row':>3}  {'config hash':32}  {'clean':>7}  "
        f"{'certified':>9}  {'adversarial':>11}"
    )
    for index, model in enumerate(front):
        result = complete_evaluation(model)
        print(
            f"{index:>3}  {model['config_hash']}  "
            f"{result['clean_accuracy']:>6.2f}%  "
            f"{result['certified_accuracy']:>8.2f}%  "
            f"{result['adversarial_accuracy']:>10.2f}%"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", default="cnn7")
    parser.add_argument("--epsilon", type=parse_epsilon, default=2 / 255)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--index",
        type=int,
        help=(
            "Row to load after sorting by reported complete certified "
            "accuracy (default: 0)."
        ),
    )
    selection.add_argument("--config-hash", help="Load this front member.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--test-samples",
        type=int,
        default=10_000,
        help="Number of CIFAR-10 test examples to evaluate (default: all 10000).",
    )
    parser.add_argument(
        "--eval-method",
        choices=("IBP", "CROWN-IBP", "ADAPTIVE"),
        default="ADAPTIVE",
        help="Incomplete certification method used by CTRAIN.evaluate().",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Evaluation device (default: CUDA when available, otherwise CPU).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("CTRAIN_DATA_ROOT", REPO_ROOT / "data")),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print the front without downloading or evaluating a checkpoint.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.test_samples <= 0:
        raise ValueError("--test-samples must be positive")

    front = list_models(
        dataset="cifar10",
        architecture=args.architecture,
        method="mtl_ibp",
        epsilon=args.epsilon,
    )
    if not front:
        raise LookupError(
            "No published CIFAR-10 MTL-IBP models match architecture="
            f"{args.architecture!r}, epsilon={args.epsilon}"
        )
    front.sort(
        key=lambda model: (
            complete_evaluation(model)["certified_accuracy"],
            complete_evaluation(model)["clean_accuracy"],
        ),
        reverse=True,
    )
    print_front(front)
    if args.list_only:
        return 0

    if args.config_hash is not None:
        matches = [
            model for model in front if model["config_hash"] == args.config_hash
        ]
        if not matches:
            raise ValueError(f"Config hash is not on this front: {args.config_hash}")
        selected = matches[0]
    else:
        index = 0 if args.index is None else args.index
        if not -len(front) <= index < len(front):
            raise IndexError(
                f"--index {index} is outside the front's {len(front)} rows"
            )
        selected = front[index]

    import torch
    from CTRAIN.data_loaders import load_cifar10

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    reported = complete_evaluation(selected)
    print(
        f"\nLoading row {front.index(selected)}: {selected['model_id']} "
        f"on {device}"
    )
    model = load_model(
        model_id=selected["model_id"],
        device=device,
        cache_dir=args.cache_dir,
    )
    _, test_loader = load_cifar10(
        batch_size=args.batch_size,
        data_root=str(args.data_root),
        val_split=False,
    )

    clean, certified, adversarial = model.evaluate(
        test_loader,
        test_samples=min(args.test_samples, len(test_loader.dataset)),
        eval_method=args.eval_method,
    )

    print("\nManifest result (complete verification)")
    print(f"  clean:       {reported['clean_accuracy']:.2f}%")
    print(f"  certified:   {reported['certified_accuracy']:.2f}%")
    print(f"  adversarial: {reported['adversarial_accuracy']:.2f}%")
    print(f"\nFresh CTRAIN evaluation ({args.eval_method} certification + PGD)")
    print(f"  clean:       {100 * clean:.2f}%")
    print(f"  certified:   {100 * certified:.2f}%")
    print(f"  adversarial: {100 * adversarial:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
