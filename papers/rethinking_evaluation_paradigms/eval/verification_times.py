"""Generate verification-time statistics from the archived raw attempts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1]
VERIFICATION_ROOT = PAPER_ROOT / "results" / "verification" / "main"
CRASHED_RUNNING_TIME = 10_000_000_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the archived verification-time table."
    )
    parser.add_argument(
        "--main-summary",
        type=Path,
        default=VERIFICATION_ROOT / "summary_results.json",
    )
    parser.add_argument(
        "--architecture-summary",
        type=Path,
        default=(
            VERIFICATION_ROOT
            / "summary_results_timeout300_testsamples1000.json"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PAPER_ROOT / "tables" / "verification_times" / "verification_times",
    )
    return parser.parse_args()


def raw_path(summary_path: str) -> Path:
    path = Path(summary_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "..":
        return (PAPER_ROOT / path).resolve()
    return (PAPER_ROOT / path).resolve()


def timeout_for(dataset: str, architecture: str) -> int:
    if architecture == "cnn7" and dataset in {"cifar10", "tinyimagenet"}:
        return 1000
    return 300


def selected_rows(main_summary: Path, architecture_summary: Path):
    with main_summary.open() as handle:
        main = json.load(handle)
    with architecture_summary.open() as handle:
        architecture = json.load(handle)
    def is_main_benchmark(row):
        return row["architecture"] == "cnn7" and row["dataset"] in {
            "cifar10",
            "tinyimagenet",
        }

    # CIFAR-10/Tiny ImageNet CNN7 use the 10,000-sample, 1,000-second summary.
    # MNIST and architecture comparisons use the 300-second appendix summary.
    return [
        *(row for row in main if is_main_benchmark(row)),
        *(row for row in architecture if not is_main_benchmark(row)),
    ]


def attempt_times(path: Path, timeout: int, test_samples: int):
    with path.open() as handle:
        attempts = json.load(handle)
    if not isinstance(attempts, dict):
        raise ValueError(f"Expected an index-keyed object in {path}")

    times = []
    crash_count = 0
    indexed_attempts = []
    for index, attempt in attempts.items():
        try:
            numeric_index = int(index)
        except ValueError as exc:
            raise ValueError(f"Non-integer test index {index!r} in {path}") from exc
        indexed_attempts.append((numeric_index, index, attempt))

    for _, index, attempt in sorted(indexed_attempts)[:test_samples]:
        if not isinstance(attempt, dict) or "running_time" not in attempt:
            raise ValueError(f"Missing running_time for index {index} in {path}")
        running_time = attempt["running_time"]
        if not isinstance(running_time, (int, float)) or not math.isfinite(running_time):
            raise ValueError(f"Invalid running_time at index {index} in {path}")
        if running_time < 0:
            raise ValueError(f"Negative running_time at index {index} in {path}")
        if running_time == CRASHED_RUNNING_TIME or running_time >= 1e9:
            crash_count += 1
            running_time = timeout
        times.append(min(float(running_time), timeout))
    return times, crash_count


def aggregate(rows):
    groups = defaultdict(
        lambda: {"times": [], "configs": 0, "crashes": 0, "missing": 0}
    )
    seen_hashes = set()
    for row in rows:
        identity = (
            row["dataset"],
            row["architecture"],
            float(row["eps"]),
            row["cert_train_method"],
            row["hash"],
        )
        if identity in seen_hashes:
            raise ValueError(f"Configuration appears in both summaries: {identity}")
        seen_hashes.add(identity)
        key = identity[:4]
        timeout = timeout_for(row["dataset"], row["architecture"])
        test_samples = int(row["total_samples"])
        times, crashes = attempt_times(
            raw_path(row["file"]), timeout, test_samples
        )
        groups[key]["times"].extend(times)
        groups[key]["configs"] += 1
        groups[key]["crashes"] += crashes
        groups[key]["missing"] += max(0, test_samples - len(times))

    output = []
    for key, values in sorted(groups.items()):
        dataset, architecture, eps, method = key
        times = values["times"]
        output.append(
            {
                "dataset": dataset,
                "architecture": architecture,
                "eps": eps,
                "method": method,
                "timeout_s": timeout_for(dataset, architecture),
                "configurations": values["configs"],
                "observed_instances": len(times),
                "missing_instances": values["missing"],
                "crashed_instances": values["crashes"],
                "average_time_s": sum(times) / len(times),
                "total_time_h": sum(times) / 3600,
            }
        )
    return output


def write_outputs(rows, output_prefix: Path):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    method_labels = {
        "mtl_ibp": "MTL-IBP",
        "sabr": "SABR",
        "shi": "IBP",
        "crown_ibp": "CROWN-IBP",
        "crown_ibp_nofusion": "CROWN-IBP (no fusion)",
    }
    lines = [
        "\\begin{tabular}{llllrrrr}",
        "\\toprule",
        "Dataset & Architecture & $\\epsilon$ & Method & Mean (s) & Total (h) & Configs & Crashes \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & {row['architecture']} & {row['eps']:.4f} & "
            f"{method_labels[row['method']]} & {row['average_time_s']:.2f} & "
            f"{row['total_time_h']:.2f} & {row['configurations']} & "
            f"{row['crashed_instances']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    output_prefix.with_suffix(".tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path} and {output_prefix.with_suffix('.tex')}")


def main():
    args = parse_args()
    rows = aggregate(selected_rows(args.main_summary, args.architecture_summary))
    if not rows:
        raise ValueError("No verification-time rows were produced")
    write_outputs(rows, args.output_prefix)


if __name__ == "__main__":
    main()
