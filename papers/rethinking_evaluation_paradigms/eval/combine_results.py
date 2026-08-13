import argparse
import json
from pathlib import Path

import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERIFICATION_ROOT = PAPER_ROOT / "results" / "verification" / "main"
DEFAULT_CLEAN_ROOT = PAPER_ROOT / "results" / "verification" / "clean_accuracy"
CRASHED_RUNNING_TIME = 10_000_000_000


def summary_filename(test_samples, artificial_timeout):
    name = "summary_results"
    if artificial_timeout != 1000:
        timeout_label = f"{artificial_timeout:g}"
        name += f"_timeout{timeout_label}"
    if test_samples != 10_000:
        name += f"_testsamples{test_samples}"
    return f"{name}.json"


def selected_indices(data, test_samples, selection, random_seed):
    available = sorted(int(index) for index in data)
    sample_count = min(test_samples, len(available))
    if selection == "first":
        return set(available[:sample_count])
    # RandomState preserves the selection produced by the publication script.
    rng = np.random.RandomState(random_seed)
    return set(rng.choice(available, size=sample_count, replace=False).tolist())


def parse_result_location(file_path, verification_root):
    try:
        relative_parts = file_path.relative_to(verification_root).parts
    except ValueError as exc:
        raise ValueError(
            f"Result file is outside verification root: {file_path}"
        ) from exc
    if len(relative_parts) != 6:
        raise ValueError(
            "Expected verification results at "
            "<dataset>/<architecture>/<eps>/<method>/<hash>/results.json; "
            f"got {file_path}"
        )
    dataset, architecture, eps, method, config_hash, filename = relative_parts
    if filename != "results.json":
        raise ValueError(f"Expected results.json, got {filename}")
    return dataset, architecture, eps, method, config_hash


def parse_results(
    verification_root=DEFAULT_VERIFICATION_ROOT,
    clean_root=DEFAULT_CLEAN_ROOT,
    test_samples=10_000,
    artificial_timeout=1000,
    selection="first",
    random_seed=42,
):
    verification_root = Path(verification_root).resolve()
    clean_root = Path(clean_root).resolve()
    result_files = sorted(verification_root.glob("**/results.json"))
    if not result_files:
        raise FileNotFoundError(
            f"No results.json files found under {verification_root}"
        )

    results = []
    test_indices = None

    for file_path in result_files:
        with file_path.open() as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            print(f"Skipping {file_path}: expected a JSON object keyed by test index")
            continue

        if test_indices is None:
            available_indices = {
                int(index)
                for index in data
                if str(index).lstrip("-").isdigit()
            }
            if len(available_indices) < test_samples:
                print(
                    f"Skipping {file_path}: only {len(available_indices)} indices "
                    f"are present; need {test_samples} to initialise selection."
                )
                continue
            test_indices = selected_indices(
                data, test_samples, selection, random_seed
            )

        try:
            dataset, architecture, eps, method, config_hash = parse_result_location(
                file_path, verification_root
            )
        except ValueError as exc:
            print(f"Skipping {file_path}: {exc}")
            continue

        # Timeout evidence must be computed from the complete raw result, not
        # from a subsample; otherwise changing test_samples can change which
        # configurations are admitted to the summary.
        valid_running_times = [
            item.get("running_time")
            for item in data.values()
            if isinstance(item, dict)
            and item.get("running_time") is not None
            and item.get("running_time") != CRASHED_RUNNING_TIME
        ]
        max_running_time = max(valid_running_times, default=0)

        unsat = sat = unknown = misclassified = error = 0
        malformed = False
        for index, item in data.items():
            try:
                int_index = int(index)
            except (TypeError, ValueError):
                print(f"Skipping {file_path}: non-integer test index {index!r}")
                malformed = True
                break
            if int_index not in test_indices:
                continue
            if (
                not isinstance(item, dict)
                or "result" not in item
                or "running_time" not in item
            ):
                print(f"Skipping {file_path}: malformed result at test index {index}")
                malformed = True
                break

            status = item["result"]
            running_time = item["running_time"]
            if (
                status is None
                or running_time is None
                or running_time > artificial_timeout
            ):
                unknown += 1
            elif status == "unsat":
                unsat += 1
            elif status == "sat":
                sat += 1
                if item.get("method") == "clean_classification":
                    misclassified += 1
            elif status == "timeout":
                unknown += 1
            elif status == "unknown":
                error += 1
                unknown += 1
            else:
                print(
                    f"Skipping {file_path}: unknown status {status!r} "
                    f"at test index {index}"
                )
                malformed = True
                break
        if malformed:
            continue

        if np.isfinite(artificial_timeout):
            tolerance = artificial_timeout * 0.1
            if max_running_time < artificial_timeout - tolerance:
                print(
                    f"Skipping {file_path}: maximum raw running time "
                    f"({max_running_time:.2f}s) is lower than the requested cap "
                    f"({artificial_timeout}s); the source run may have used a "
                    "shorter timeout."
                )
                continue

        total_samples = unsat + sat + unknown
        if total_samples < test_samples:
            print(
                f"Skipping {file_path}: only {total_samples} selected samples "
                "are present; "
                f"expected {test_samples}."
            )
            continue

        clean_path = clean_root / (
            f"{dataset}_{architecture}_{method}{eps}_{config_hash}_nat_acc.json"
        )
        if not clean_path.exists():
            print(f"Skipping {file_path}: missing clean-accuracy result {clean_path}")
            continue
        with clean_path.open() as handle:
            clean_data = json.load(handle)
        if "std_acc" not in clean_data:
            print(f"Skipping {file_path}: {clean_path} has no std_acc value")
            continue

        results.append(
            {
                "file": str(file_path.relative_to(PAPER_ROOT)),
                "dataset": dataset,
                "architecture": architecture,
                "eps": eps,
                "cert_train_method": method,
                "hash": config_hash,
                "total_samples": total_samples,
                "unsat": unsat,
                "sat": sat,
                "unknown": unknown,
                "misclassified": misclassified,
                "error": error,
                "adversarial_accuracy": round(
                    (total_samples - sat) / total_samples * 100, 2
                ),
                "certified_accuracy": round(unsat / total_samples * 100, 2),
                "clean_classification_accuracy": round(clean_data["std_acc"] * 100, 2),
            }
        )

    return sorted(
        results,
        key=lambda item: (
            item["dataset"],
            item["architecture"],
            float(item["eps"]),
            item["cert_train_method"],
            item["hash"],
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine raw complete-verification and clean-accuracy results."
    )
    parser.add_argument(
        "--verification-root", type=Path, default=DEFAULT_VERIFICATION_ROOT
    )
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--test-samples", type=int, default=10_000)
    parser.add_argument("--artificial-timeout", type=float, default=1000)
    parser.add_argument(
        "--sample-selection", choices=["first", "random"], default="first"
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.test_samples <= 0:
        raise ValueError("--test-samples must be positive")
    if args.artificial_timeout <= 0:
        raise ValueError("--artificial-timeout must be positive")

    results = parse_results(
        verification_root=args.verification_root,
        clean_root=args.clean_root,
        test_samples=args.test_samples,
        artificial_timeout=args.artificial_timeout,
        selection=args.sample_selection,
        random_seed=args.random_seed,
    )
    output_path = args.output or (
        args.verification_root
        / summary_filename(args.test_samples, args.artificial_timeout)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(results, handle, indent=4)
        handle.write("\n")
    print(f"Wrote {len(results)} configurations to {output_path}")


if __name__ == "__main__":
    main()
