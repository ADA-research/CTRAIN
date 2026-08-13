"""Plot incomplete fronts for expressive-loss CC-IBP and Exp-IBP studies.

This script is intentionally separate from validation-vs-test tuning analysis.
It reads the expressive-loss result root for CC-IBP / Exp-IBP and optionally
overlays MTL-IBP from the standard HPO root for direct comparison.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from optuna_utils import find_studies, load_trial_points
from util import hypervolume_2d, pareto_front as generic_pareto_front


PAPER_ROOT = Path(__file__).resolve().parents[1]
EXPRESSIVE_LOSSES_ROOT = Path(
    os.environ.get(
        "CTRAIN_EXPRESSIVE_LOSSES_HPO_ROOT",
        PAPER_ROOT / "results" / "hpo" / "expressive_losses",
    )
)
MTL_ROOT = Path(
    os.environ.get(
        "CTRAIN_TEST_HPO_ROOT",
        PAPER_ROOT / "results" / "hpo" / "main" / "optuna_studies",
    )
)
OUTPUT_DIR = PAPER_ROOT / "results" / "analysis" / "expressive_losses"
PLOTS_DIR = PAPER_ROOT / "plots" / "expressive_losses"
PLOT_FORMATS = ("pdf",)
KNOWN_DATASETS = ("cifar10", "tinyimagenet", "mnist")
KNOWN_METHODS = ("mtl_ibp", "cc_ibp", "exp_ibp")


@dataclass(frozen=True)
class Point:
    source: str
    dataset: str
    architecture: str
    method: str
    eps: str
    seed: int | None
    trial_number: int | None
    clean_acc: float
    certified_acc: float
    feasible: bool
    constraints: str | None
    path: str

    @property
    def group_key(self) -> tuple[str, str, str, str]:
        return (self.dataset, self.architecture, self.method, self.eps)


def load_optuna_points(spec, source: str, max_trials: int | None) -> list[Point]:
    return [
        Point(
            source=source,
            dataset=spec.dataset,
            architecture=spec.architecture,
            method=spec.method,
            eps=spec.eps,
            seed=spec.seed,
            trial_number=row["trial_number"],
            clean_acc=row["clean_acc"],
            certified_acc=row["certified_acc"],
            feasible=row["feasible"],
            constraints=row["constraints"],
            path=spec.path.name,
        )
        for row in load_trial_points(spec, max_trials)
    ]


def pareto_front(points: Iterable[Point]) -> list[Point]:
    front = generic_pareto_front(
        list(points), lambda point: (point.clean_acc, point.certified_acc)
    )
    return sorted(front, key=lambda point: (point.clean_acc, point.certified_acc))


def hypervolume(points: Iterable[Point]) -> float:
    return hypervolume_2d(
        points, lambda point: (point.clean_acc, point.certified_acc)
    )


def display_accuracy(value: float) -> float:
    return value * 100.0 if value <= 1.5 else value


def group_points(points: Iterable[Point]) -> dict[tuple[str, str, str], list[Point]]:
    groups: dict[tuple[str, str, str], list[Point]] = {}
    for point in points:
        key = (point.dataset, point.architecture, point.eps)
        groups.setdefault(key, []).append(point)
    return groups


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


def plot_fronts(
    grouped_fronts: dict[str, list[Point]],
    output_base: Path,
    formats: tuple[str, ...],
    legend: bool,
) -> list[Path]:
    mpl_config_dir = (
        Path(os.environ.get("TMPDIR", "/tmp"))
        / "ctrain_expressive_losses_matplotlib"
    )
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    method_style = {
        "mtl_ibp": ("MTL-IBP", "o", 0),
        "exp_ibp": ("Exp-IBP", "D", 1),
        "cc_ibp": ("CC-IBP", "^", 2),
    }
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(10, 10))
    palette = sns.color_palette("Set2", 6)

    plotted = []
    for method in ("mtl_ibp", "exp_ibp", "cc_ibp"):
        front = grouped_fronts.get(method, [])
        if not front:
            continue
        label, marker, color_index = method_style[method]
        ax.plot(
            [display_accuracy(point.certified_acc) for point in front],
            [display_accuracy(point.clean_acc) for point in front],
            marker=marker,
            markersize=20,
            markeredgecolor="black",
            markeredgewidth=2.5,
            linewidth=10,
            alpha=0.55,
            color=palette[color_index],
            zorder=3,
            label=label,
        )
        plotted.extend(front)

    ax.set_xlabel("Certified Accuracy", fontsize=35, fontweight="bold", labelpad=16)
    ax.set_ylabel("Natural Accuracy", fontsize=35, fontweight="bold", labelpad=16)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.tick_params(axis="both", which="major", labelsize=25, colors="black", length=8, width=2)

    if plotted:
        x_values = [display_accuracy(point.certified_acc) for point in plotted]
        y_values = [display_accuracy(point.clean_acc) for point in plotted]
        x_pad = max(2.0, (max(x_values) - min(x_values)) * 0.08)
        y_pad = max(2.0, (max(y_values) - min(y_values)) * 0.08)
        ax.set_xlim(max(0.0, min(x_values) - x_pad), min(100.0, max(x_values) + x_pad))
        ax.set_ylim(max(0.0, min(y_values) - y_pad), min(100.0, max(y_values) + y_pad))

    if legend:
        ax.legend(fontsize=30, frameon=True)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        out_path = output_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=False)
        written.append(out_path)
    plt.close(fig)
    return written


def build_summary(points: list[Point]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows = []
    front_rows = []
    for key, group in sorted(group_points(points).items()):
        dataset, architecture, eps = key
        for method in sorted({point.method for point in group}):
            method_points = [point for point in group if point.method == method]
            front = pareto_front(method_points)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "architecture": architecture,
                    "method": method,
                    "eps": eps,
                    "points": len(method_points),
                    "front_points": len(front),
                    "hypervolume": hypervolume(front),
                }
            )
            front_rows.extend(asdict(point) for point in front)
    return summary_rows, front_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CC-IBP, Exp-IBP, and MTL-IBP incomplete fronts.")
    parser.add_argument(
        "--expressive-losses-root", type=Path, default=EXPRESSIVE_LOSSES_ROOT
    )
    parser.add_argument("--mtl-root", type=Path, default=MTL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--datasets", nargs="+", default=["cifar10"])
    parser.add_argument("--architectures", nargs="+", default=["cnn7"])
    parser.add_argument("--eps", nargs="+", default=["0.00784313725490196", "0.03137254901960784"])
    parser.add_argument("--max-trials-per-study", type=int, default=100)
    parser.add_argument("--include-infeasible", action="store_true")
    parser.add_argument("--plot-formats", default=",".join(PLOT_FORMATS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_trials = args.max_trials_per_study if args.max_trials_per_study > 0 else None
    datasets = set(args.datasets)

    expressive_losses_specs = find_studies(
        args.expressive_losses_root,
        KNOWN_DATASETS,
        KNOWN_METHODS,
        selected_datasets=datasets,
        selected_methods={"cc_ibp", "exp_ibp"},
        complete=False,
    )
    mtl_specs = find_studies(
        args.mtl_root,
        KNOWN_DATASETS,
        KNOWN_METHODS,
        selected_datasets=datasets,
        selected_methods={"mtl_ibp"},
        complete=False,
    )
    specs = [
        spec for spec in [*expressive_losses_specs, *mtl_specs]
        if spec.architecture in set(args.architectures) and spec.eps in set(args.eps)
    ]
    points = [
        point
        for spec in specs
        for point in load_optuna_points(
            spec,
            "expressive_losses"
            if spec.method in {"cc_ibp", "exp_ibp"}
            else "mtl_baseline",
            max_trials,
        )
    ]
    if not args.include_infeasible:
        points = [point for point in points if point.feasible]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "expressive_losses_incomplete_points.csv",
        [asdict(point) for point in points],
    )
    summary_rows, front_rows = build_summary(points)
    write_csv(args.output_dir / "expressive_losses_front_summary.csv", summary_rows)
    write_csv(args.output_dir / "expressive_losses_front_points.csv", front_rows)

    formats = tuple(fmt.strip() for fmt in args.plot_formats.split(",") if fmt.strip())
    written = []
    for key, group in sorted(group_points(points).items()):
        dataset, architecture, eps = key
        fronts = {
            method: pareto_front([point for point in group if point.method == method])
            for method in sorted({point.method for point in group})
        }
        for method in ("cc_ibp", "exp_ibp"):
            if fronts.get(method):
                base = args.plots_dir / "separate" / f"{dataset}_{architecture}_{method}_{eps.replace('.', 'p')}_incomplete_nolegend"
                written.extend(plot_fronts({method: fronts[method]}, base, formats, legend=False))
                base = args.plots_dir / "separate" / f"{dataset}_{architecture}_{method}_{eps.replace('.', 'p')}_incomplete_legend"
                written.extend(plot_fronts({method: fronts[method]}, base, formats, legend=True))
        combined_fronts = {method: fronts.get(method, []) for method in ("mtl_ibp", "exp_ibp", "cc_ibp")}
        if all(combined_fronts.values()):
            base = args.plots_dir / "combined" / f"{dataset}_{architecture}_mtl_exp_cc_ibp_{eps.replace('.', 'p')}_incomplete_nolegend"
            written.extend(plot_fronts(combined_fronts, base, formats, legend=False))
            base = args.plots_dir / "combined" / f"{dataset}_{architecture}_mtl_exp_cc_ibp_{eps.replace('.', 'p')}_incomplete_legend"
            written.extend(plot_fronts(combined_fronts, base, formats, legend=True))

    print(
        f"Wrote {len(points)} points to {args.output_dir} and "
        f"{len(written)} plots to {args.plots_dir}"
    )


if __name__ == "__main__":
    main()
