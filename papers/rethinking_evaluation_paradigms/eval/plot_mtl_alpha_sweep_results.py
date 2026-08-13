"""Plot fixed-alpha MTL-IBP fronts against the archived MO-HPO fronts."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from pathlib import Path

from util import pareto_front

PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALPHA_ROOT = PAPER_ROOT / "results" / "hpo" / "alpha_sweep"
DEFAULT_FRONTS_ROOT = PAPER_ROOT / "results" / "hpo" / "main" / "pareto_fronts"
DEFAULT_COMPLETE_SUMMARY = (
    PAPER_ROOT / "results" / "verification" / "main" / "summary_results.json"
)
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "plots" / "alpha_sweep"
TRIAL_PATTERN = re.compile(r"^Trial\s+(\d+):\s*([\[(].*[\])])$")
HASH_PATTERN = re.compile(r"^Config hash:\s*([0-9a-f]{32})$")


def parse_eps(value: str) -> float:
    aliases = {"2/255": 2 / 255, "8/255": 8 / 255}
    return aliases[value] if value in aliases else float(value)


def eps_tag(eps: float) -> str:
    numerator = round(eps * 255)
    if abs(eps - numerator / 255) < 1e-12:
        return f"{numerator}_255"
    return f"{eps:.6g}".replace(".", "_")


def load_alpha_front(alpha_root: Path, eps: float) -> list[dict[str, float]]:
    path = (
        alpha_root
        / f"cifar10_cnn7_mtl_ibp_alpha_sweep_{eps_tag(eps)}"
        / "pareto_front.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed-alpha Pareto front: {path}")

    with path.open(newline="") as handle:
        points = [
            {
                "alpha": float(row["alpha"]),
                "natural_accuracy": float(row["nat_acc"]),
                "certified_accuracy": float(row["cert_acc"]),
            }
            for row in csv.DictReader(handle)
        ]
    if not points:
        raise ValueError(f"No points found in {path}")
    return points


def load_archived_front(path: Path) -> list[dict[str, float | int | str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing archived MTL-IBP front: {path}")

    points = []
    for line in path.read_text().splitlines():
        match = TRIAL_PATTERN.match(line)
        if match is not None:
            values = ast.literal_eval(match.group(2))
            if len(values) != 2:
                raise ValueError(f"Expected two objectives in {line!r}")
            points.append(
                {
                    "trial": int(match.group(1)),
                    "natural_accuracy": float(values[0]),
                    "certified_accuracy": float(values[1]),
                }
            )
            continue
        hash_match = HASH_PATTERN.match(line)
        if hash_match is not None and points:
            points[-1]["hash"] = hash_match.group(1)
    if not points:
        raise ValueError(f"No trial objectives found in {path}")
    if any("hash" not in point for point in points):
        raise ValueError(f"Some trials have no configuration hash in {path}")
    return points


def load_mtl_front(
    fronts_root: Path, eps: float
) -> list[dict[str, float | int | str]]:
    stem = f"pareto_front_mtl_ibp_cnn7_cifar10_{eps}"
    return load_archived_front(fronts_root / f"{stem}_subselected0.05.txt")


def load_complete_pareto_hashes(summary_path: Path, eps: float) -> set[str]:
    with summary_path.open() as handle:
        results = json.load(handle)
    results = [
        result
        for result in results
        if result["dataset"] == "cifar10"
        and result["architecture"] == "cnn7"
        and result["cert_train_method"] == "mtl_ibp"
        and abs(float(result["eps"]) - eps) < 1e-12
    ]
    if not results:
        raise ValueError(
            f"No matching complete-verification results in {summary_path}"
        )
    return {result["hash"] for result in pareto_front(results)}


def plot_comparison(
    alpha_front: list[dict[str, float]],
    mtl_front: list[dict[str, float | int | str]],
    output_base: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    mpl_config_dir = (
        Path(os.environ.get("TMPDIR", "/tmp"))
        / "ctrain_mtl_alpha_sweep_matplotlib"
    )
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LogNorm

    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(6,6))

    mtl_front = sorted(mtl_front, key=lambda point: point["certified_accuracy"])
    ax.plot(
        [100 * point["certified_accuracy"] for point in mtl_front],
        [100 * point["natural_accuracy"] for point in mtl_front],
        color="#4c72b0",
        linewidth=4,
        alpha=0.35,
        zorder=1,
    )
    ax.scatter(
        [100 * point["certified_accuracy"] for point in mtl_front],
        [100 * point["natural_accuracy"] for point in mtl_front],
        s=220,
        marker="o",
        color="#4c72b0",
        edgecolor="black",
        linewidth=1.5,
        label="Multi-Objective\nHPO (Ours)",
        zorder=4,
    )

    alpha_front = sorted(
        alpha_front, key=lambda point: point["certified_accuracy"]
    )
    alphas = [point["alpha"] for point in alpha_front]
    scatter = ax.scatter(
        [100 * point["certified_accuracy"] for point in alpha_front],
        [100 * point["natural_accuracy"] for point in alpha_front],
        c=alphas,
        norm=LogNorm(vmin=min(alphas), vmax=max(alphas)),
        cmap="viridis",
        s=130,
        marker="D",
        edgecolor="black",
        linewidth=0.8,
        label=r"$\alpha$-sweep",
        zorder=3,
    )
    ax.plot(
        [100 * point["certified_accuracy"] for point in alpha_front],
        [100 * point["natural_accuracy"] for point in alpha_front],
        color="#55a868",
        linewidth=4,
        alpha=0.7,
        zorder=2,
    )

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label(r"$\alpha$", fontsize=20)
    colorbar.ax.tick_params(labelsize=14)
    ax.set_xlabel("Certified Accuracy (%)", fontsize=25, fontweight="bold")
    ax.set_ylabel("Natural Accuracy (%)", fontsize=25, fontweight="bold")
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(fontsize=16, frameon=True)
    sns.despine(ax=ax)
    fig.tight_layout()

    output_base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        output_path = output_base.with_suffix(f".{fmt}")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        written.append(output_path)
    plt.close(fig)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-root", type=Path, default=DEFAULT_ALPHA_ROOT)
    parser.add_argument("--fronts-root", type=Path, default=DEFAULT_FRONTS_ROOT)
    parser.add_argument(
        "--complete-summary", type=Path, default=DEFAULT_COMPLETE_SUMMARY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eps", nargs="+", default=["2/255", "8/255"])
    parser.add_argument("--plot-formats", default="pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = tuple(
        fmt.strip() for fmt in args.plot_formats.split(",") if fmt.strip()
    )
    if not formats:
        raise ValueError("--plot-formats must contain at least one format")

    written = []
    for eps_value in args.eps:
        eps = parse_eps(eps_value)
        complete_pareto_hashes = load_complete_pareto_hashes(
            args.complete_summary, eps
        )
        mtl_front = [
            point
            for point in load_mtl_front(args.fronts_root, eps)
            if point["hash"] in complete_pareto_hashes
        ]
        if not mtl_front:
            raise ValueError(
                f"No complete-verification Pareto points matched eps={eps}"
            )
        written.extend(
            plot_comparison(
                load_alpha_front(args.alpha_root, eps),
                mtl_front,
                args.output_dir / f"mtl_alpha_sweep_vs_mo_hpo_{eps_tag(eps)}",
                formats,
            )
        )

    print(f"Wrote {len(written)} plots to {args.output_dir}")


if __name__ == "__main__":
    main()
