"""Compare validation-split tuning against test-set tuning.

The incomplete-verification analysis reads Optuna SQLite studies directly and
compares the Pareto fronts over natural and certified accuracy. Complete
verification can be analysed with the same code once summary JSON files are
available for both tuning regimes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from optuna_utils import find_studies as find_archived_studies, load_trial_points
from util import dominates as dominates_values
from util import hypervolume_2d, pareto_front as generic_pareto_front


PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_ROOT = Path(
    os.environ.get(
        "CTRAIN_VALIDATION_HPO_ROOT",
        PAPER_ROOT / "results" / "hpo" / "validation" / "optuna_studies",
    )
)
DEFAULT_TEST_ROOT = Path(
    os.environ.get(
        "CTRAIN_TEST_HPO_ROOT",
        PAPER_ROOT / "results" / "hpo" / "main" / "optuna_studies",
    )
)
DEFAULT_OUTPUT_ROOT = PAPER_ROOT / "results" / "analysis" / "validation_vs_test"
DEFAULT_PLOTS_DIR = PAPER_ROOT / "plots" / "validation_vs_test"
DEFAULT_PLOT_FORMATS = ("pdf",)

KNOWN_DATASETS = ("cifar10", "tinyimagenet", "mnist")
KNOWN_METHODS = (
    "crown_ibp_nofusion",
    "crown_ibp",
    "mtl_ibp",
    "sabr",
    "shi",
)


@dataclass(frozen=True)
class Point:
    source: str
    verification: str
    dataset: str
    architecture: str
    method: str
    eps: str
    seed: int | None
    trial_number: int | None
    config_hash: str | None
    clean_acc: float
    certified_acc: float
    feasible: bool
    constraints: str | None
    path: str

    @property
    def coords(self) -> tuple[float, float]:
        return (self.clean_acc, self.certified_acc)

    @property
    def group_key(self) -> tuple[str, str, str, str]:
        return (self.dataset, self.architecture, self.method, self.eps)


def find_studies(root: Path, dataset: str, complete: bool | None):
    return find_archived_studies(
        root,
        KNOWN_DATASETS,
        KNOWN_METHODS,
        selected_datasets={dataset},
        complete=complete,
    )


def load_optuna_points(spec, source: str, max_trials: int | None) -> list[Point]:
    return [
        Point(
            source=source,
            verification="incomplete",
            dataset=spec.dataset,
            architecture=spec.architecture,
            method=spec.method,
            eps=spec.eps,
            seed=spec.seed,
            trial_number=row["trial_number"],
            config_hash=None,
            clean_acc=row["clean_acc"],
            certified_acc=row["certified_acc"],
            feasible=row["feasible"],
            constraints=row["constraints"],
            path=spec.path.name,
        )
        for row in load_trial_points(spec, max_trials)
    ]


def load_complete_summary(path: Path, source: str, dataset: str) -> list[Point]:
    if not path.exists():
        return []
    with path.open() as handle:
        records = json.load(handle)

    points = []
    for record in records:
        if record.get("dataset") != dataset:
            continue
        points.append(
            Point(
                source=source,
                verification="complete",
                dataset=record["dataset"],
                architecture=record["architecture"],
                method=record["cert_train_method"],
                eps=str(record["eps"]),
                seed=None,
                trial_number=None,
                config_hash=record.get("hash"),
                clean_acc=float(record["clean_classification_accuracy"]),
                certified_acc=float(record["certified_accuracy"]),
                feasible=True,
                constraints=None,
                path=str(record.get("file", path)),
            )
        )
    return points


def dominates(left: Point, right: Point) -> bool:
    return dominates_values(left.coords, right.coords)


def pareto_front(points: Iterable[Point]) -> list[Point]:
    front = generic_pareto_front(list(points), lambda point: point.coords)
    return sorted(front, key=lambda p: (p.clean_acc, p.certified_acc))


def hypervolume(points: Iterable[Point], reference: tuple[float, float] = (0.0, 0.0)) -> float:
    return hypervolume_2d(points, lambda point: point.coords, reference)


def count_dominated(points: Iterable[Point], dominators: Iterable[Point]) -> int:
    dominator_items = list(dominators)
    return sum(any(dominates(other, point) for other in dominator_items) for point in points)


def group_points(points: Iterable[Point]) -> dict[tuple[str, str, str, str], list[Point]]:
    groups: dict[tuple[str, str, str, str], list[Point]] = {}
    for point in points:
        groups.setdefault(point.group_key, []).append(point)
    return groups


def compare_groups(
    validation_points: list[Point],
    test_points: list[Point],
    verification: str,
    include_unmatched: bool,
) -> tuple[list[dict[str, object]], list[Point]]:
    rows = []
    combined_front_points = []
    validation_groups = group_points(validation_points)
    test_groups = group_points(test_points)

    keys = set(validation_groups) | set(test_groups) if include_unmatched else set(validation_groups) & set(test_groups)
    for key in sorted(keys):
        val = validation_groups.get(key, [])
        test = test_groups.get(key, [])
        val_front = pareto_front(val)
        test_front = pareto_front(test)
        union_front = pareto_front([*val, *test])
        combined_front_points.extend(union_front)

        dataset, architecture, method, eps = key
        hv_val = hypervolume(val_front)
        hv_test = hypervolume(test_front)
        hv_union = hypervolume(union_front)
        rows.append(
            {
                "verification": verification,
                "dataset": dataset,
                "architecture": architecture,
                "method": method,
                "eps": eps,
                "validation_points": len(val),
                "test_points": len(test),
                "validation_front_points": len(val_front),
                "test_front_points": len(test_front),
                "combined_front_points": len(union_front),
                "combined_front_validation_points": sum(p.source == "validation" for p in union_front),
                "combined_front_test_points": sum(p.source == "test" for p in union_front),
                "validation_hypervolume": hv_val,
                "test_hypervolume": hv_test,
                "combined_hypervolume": hv_union,
                "validation_minus_test_hypervolume": hv_val - hv_test,
                "validation_hypervolume_ratio": hv_val / hv_test if hv_test else None,
                "validation_points_dominated_by_test_front": count_dominated(val, test_front),
                "test_points_dominated_by_validation_front": count_dominated(test, val_front),
                "validation_front_points_dominated_by_test_front": count_dominated(val_front, test_front),
                "test_front_points_dominated_by_validation_front": count_dominated(test_front, val_front),
            }
        )
    return rows, combined_front_points


def combined_front_analysis(points: list[Point], source: str) -> tuple[list[dict[str, object]], list[Point]]:
    source_points = [point for point in points if point.source == source]
    groups: dict[tuple[str, str, str, str], list[Point]] = {}
    for point in source_points:
        key = (point.verification, point.dataset, point.architecture, point.eps)
        groups.setdefault(key, []).append(point)

    rows = []
    front_points = []
    for key in sorted(groups):
        verification, dataset, architecture, eps = key
        front = pareto_front(groups[key])
        front_points.extend(front)
        method_counts: dict[str, int] = {}
        for point in front:
            method_counts[point.method] = method_counts.get(point.method, 0) + 1

        row = {
            "source": source,
            "verification": verification,
            "dataset": dataset,
            "architecture": architecture,
            "eps": eps,
            "total_points": len(groups[key]),
            "combined_front_points": len(front),
            "combined_hypervolume": hypervolume(front),
        }
        for method, count in sorted(method_counts.items()):
            row[f"{method}_front_points"] = count
        rows.append(row)
    return rows, front_points


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def point_rows(points: Iterable[Point]) -> list[dict[str, object]]:
    return [asdict(point) for point in points]


def eps_label(eps: str) -> str:
    try:
        value = float(eps)
    except ValueError:
        return eps
    if abs(value - 2 / 255) < 1e-10:
        return "2/255"
    if abs(value - 8 / 255) < 1e-10:
        return "8/255"
    if abs(value - 1 / 255) < 1e-10:
        return "1/255"
    return eps


def display_accuracy(value: float) -> float:
    return value * 100.0 if value <= 1.5 else value


def plot_groups(
    validation_points: list[Point],
    test_points: list[Point],
    verification: str,
    plots_dir: Path,
    include_unmatched: bool,
    formats: tuple[str, ...],
) -> list[Path]:
    mpl_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "ctrain_validation_vs_test_tuning_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    validation_groups = group_points(validation_points)
    test_groups = group_points(test_points)
    keys = set(validation_groups) | set(test_groups) if include_unmatched else set(validation_groups) & set(test_groups)
    plot_dir = plots_dir / verification
    plot_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for key in sorted(keys):
        val = validation_groups.get(key, [])
        test = test_groups.get(key, [])
        if not val and not test:
            continue

        val_front = pareto_front(val)
        test_front = pareto_front(test)
        dataset, architecture, method, eps = key

        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=(10, 10))
        palette = sns.color_palette("Set2", 4)
        if test_front:
            ax.plot(
                [display_accuracy(p.certified_acc) for p in test_front],
                [display_accuracy(p.clean_acc) for p in test_front],
                marker="o",
                markersize=20,
                markeredgecolor="black",
                markeredgewidth=2.5,
                linewidth=10,
                alpha=0.55,
                color=palette[0],
                zorder=3,
                label="Tuned on test set",
            )
        if val_front:
            ax.plot(
                [display_accuracy(p.certified_acc) for p in val_front],
                [display_accuracy(p.clean_acc) for p in val_front],
                marker="s",
                markersize=20,
                markeredgecolor="black",
                markeredgewidth=2.5,
                linewidth=10,
                alpha=0.55,
                color=palette[1],
                zorder=3,
                label="Tuned on validation set",
            )

        ax.set_xlabel("Certified Accuracy", fontsize=35, fontweight="bold", labelpad=16)
        ax.set_ylabel("Natural Accuracy", fontsize=35, fontweight="bold", labelpad=16)
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        ax.tick_params(axis="both", which="major", labelsize=25, colors="black", length=8, width=2)

        plotted_points = [
            *val_front,
            *test_front,
        ]
        x_values = [display_accuracy(p.certified_acc) for p in plotted_points]
        y_values = [display_accuracy(p.clean_acc) for p in plotted_points]
        if x_values and y_values:
            x_pad = max(2.0, (max(x_values) - min(x_values)) * 0.08)
            y_pad = max(2.0, (max(y_values) - min(y_values)) * 0.08)
            ax.set_xlim(max(0.0, min(x_values) - x_pad), min(100.0, max(x_values) + x_pad))
            ax.set_ylim(max(0.0, min(y_values) - y_pad), min(100.0, max(y_values) + y_pad))

        fig.tight_layout()
        base_name = f"{dataset}_{architecture}_{method}_{eps.replace('.', 'p')}_{verification}"
        for fmt in formats:
            legend_path = plot_dir / f"{base_name}_legend.{fmt}"
            ax.legend(fontsize=30, frameon=True)
            fig.savefig(legend_path, dpi=300, bbox_inches="tight", transparent=False)
            written.append(legend_path)

            nolegend_path = plot_dir / f"{base_name}_nolegend.{fmt}"
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            fig.savefig(nolegend_path, dpi=300, bbox_inches="tight", transparent=False)
            written.append(nolegend_path)
        plt.close(fig)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse validation-split tuning vs test-set tuning."
    )
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis-data directory (default: grouped by verification mode and dataset).",
    )
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument(
        "--max-trials-per-study",
        type=int,
        default=100,
        help="Cap Optuna trials per seed for incomplete verification; use 0 for no cap.",
    )
    parser.add_argument("--skip-incomplete", action="store_true")
    parser.add_argument(
        "--include-unmatched",
        action="store_true",
        help="Also emit groups that exist in only one tuning root.",
    )
    parser.add_argument(
        "--include-infeasible",
        action="store_true",
        help="Include Optuna trials with positive constraint violations in fronts, hypervolume, and plots.",
    )
    parser.add_argument("--validation-complete-summary", type=Path, default=None)
    parser.add_argument("--test-complete-summary", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--plot-formats",
        default=",".join(DEFAULT_PLOT_FORMATS),
        help="Comma-separated plot formats, e.g. pdf,png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        has_complete = bool(
            args.validation_complete_summary and args.test_complete_summary
        )
        if args.skip_incomplete and has_complete:
            verification_dir = "complete"
        elif not args.skip_incomplete and not has_complete:
            verification_dir = "incomplete"
        else:
            verification_dir = "combined"
        args.output_dir = DEFAULT_OUTPUT_ROOT / verification_dir / args.dataset
    max_trials = args.max_trials_per_study if args.max_trials_per_study > 0 else None
    all_rows = []
    all_fronts = []
    validation_combined_rows = []
    validation_combined_fronts = []
    all_plots = []
    plot_formats = tuple(fmt.strip() for fmt in args.plot_formats.split(",") if fmt.strip())

    if not args.skip_incomplete:
        validation_specs = find_studies(args.validation_root, args.dataset, complete=False)
        test_specs = find_studies(args.test_root, args.dataset, complete=False)
        validation_points = [
            point
            for spec in validation_specs
            for point in load_optuna_points(spec, "validation", max_trials)
        ]
        test_points = [
            point
            for spec in test_specs
            for point in load_optuna_points(spec, "test", max_trials)
        ]
        if not args.include_infeasible:
            validation_points = [point for point in validation_points if point.feasible]
            test_points = [point for point in test_points if point.feasible]
        rows, fronts = compare_groups(
            validation_points,
            test_points,
            "incomplete",
            include_unmatched=args.include_unmatched,
        )
        all_rows.extend(rows)
        all_fronts.extend(fronts)
        combined_rows, combined_fronts = combined_front_analysis(validation_points, "validation")
        validation_combined_rows.extend(combined_rows)
        validation_combined_fronts.extend(combined_fronts)
        write_csv(args.output_dir / "validation_vs_test_tuning_incomplete_points.csv", point_rows(validation_points + test_points))
        if not args.no_plots:
            all_plots.extend(
                plot_groups(
                    validation_points,
                    test_points,
                    "incomplete",
                    args.plots_dir,
                    include_unmatched=args.include_unmatched,
                    formats=plot_formats,
                )
            )

    if args.validation_complete_summary and args.test_complete_summary:
        validation_points = load_complete_summary(args.validation_complete_summary, "validation", args.dataset)
        test_points = load_complete_summary(args.test_complete_summary, "test", args.dataset)
        if not args.include_infeasible:
            validation_points = [point for point in validation_points if point.feasible]
            test_points = [point for point in test_points if point.feasible]
        rows, fronts = compare_groups(
            validation_points,
            test_points,
            "complete",
            include_unmatched=args.include_unmatched,
        )
        all_rows.extend(rows)
        all_fronts.extend(fronts)
        combined_rows, combined_fronts = combined_front_analysis(validation_points, "validation")
        validation_combined_rows.extend(combined_rows)
        validation_combined_fronts.extend(combined_fronts)
        write_csv(args.output_dir / "validation_vs_test_tuning_complete_points.csv", point_rows(validation_points + test_points))
        if not args.no_plots:
            all_plots.extend(
                plot_groups(
                    validation_points,
                    test_points,
                    "complete",
                    args.plots_dir,
                    include_unmatched=args.include_unmatched,
                    formats=plot_formats,
                )
            )

    write_csv(args.output_dir / "validation_vs_test_tuning_summary.csv", all_rows)
    write_csv(args.output_dir / "validation_vs_test_tuning_combined_fronts.csv", point_rows(all_fronts))
    write_csv(args.output_dir / "validation_tuned_combined_front_summary.csv", validation_combined_rows)
    write_csv(args.output_dir / "validation_tuned_combined_front_points.csv", point_rows(validation_combined_fronts))
    with (args.output_dir / "validation_vs_test_tuning_summary.json").open("w") as handle:
        json.dump(all_rows, handle, indent=2)

    print(f"Wrote {len(all_rows)} comparison rows to {args.output_dir}")
    if all_plots:
        print(f"Wrote {len(all_plots)} plots to {args.plots_dir}")


if __name__ == "__main__":
    main()
