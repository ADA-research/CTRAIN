import argparse
import json
from pathlib import Path

from util import pareto_front


PAPER_ROOT = Path(__file__).resolve().parents[1]


def common_sample_count(group_results):
    sample_counts = {result["total_samples"] for result in group_results}
    if len(sample_counts) != 1:
        print("Warning: different total_samples in this group; skipping analysis.")
        return False
    return True


def count_pareto_front_methods(results):
    groups = {}
    for result in results:
        key = (result["dataset"], result["architecture"], result["eps"])
        groups.setdefault(key, []).append(result)

    for (dataset, architecture, eps), group_results in sorted(groups.items()):
        print(f"\nPareto front for {dataset}, {architecture}, eps={eps}:")
        if not common_sample_count(group_results):
            continue

        group_front = pareto_front(group_results)
        method_counts = {}
        for result in group_front:
            method = result["cert_train_method"]
            method_counts[method] = method_counts.get(method, 0) + 1

        print("Method contributions:")
        for method, count in sorted(method_counts.items()):
            print(f"{method}: {count} configurations")

        print("Detailed configurations:")
        for result in sorted(
            group_front,
            key=lambda item: (
                -item["clean_classification_accuracy"],
                -item["certified_accuracy"],
            ),
        ):
            print(
                f"{result['cert_train_method']}: clean="
                f"{result['clean_classification_accuracy']:.2f}%, certified="
                f"{result['certified_accuracy']:.2f}%, adversarial="
                f"{result['adversarial_accuracy']:.2f}%"
            )


def analyze_network_contributions(results):
    groups = {}
    for result in results:
        key = (result["dataset"], result["cert_train_method"], result["eps"])
        groups.setdefault(key, []).append(result)

    for (dataset, method, eps), group_results in sorted(groups.items()):
        architectures = {result["architecture"] for result in group_results}
        if len(architectures) == 1:
            continue

        print(f"\nArchitecture front for {dataset}, {method}, eps={eps}:")
        if not common_sample_count(group_results):
            continue

        group_front = pareto_front(group_results)
        architecture_counts = {}
        for result in group_front:
            architecture = result["architecture"]
            architecture_counts[architecture] = (
                architecture_counts.get(architecture, 0) + 1
            )

        for architecture, count in sorted(architecture_counts.items()):
            print(
                f"{architecture}: {count} configurations "
                f"({count / len(group_front) * 100:.1f}% of Pareto front)"
            )

        print("Detailed configurations:")
        for result in sorted(
            group_front,
            key=lambda item: (
                -item["clean_classification_accuracy"],
                -item["certified_accuracy"],
            ),
        ):
            print(
                f"{result['architecture']}: clean="
                f"{result['clean_classification_accuracy']:.2f}%, certified="
                f"{result['certified_accuracy']:.2f}%, adversarial="
                f"{result['adversarial_accuracy']:.2f}%"
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Report method and architecture contributions to combined Pareto fronts."
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PAPER_ROOT / "results" / "verification" / "main" / "summary_results.json",
    )
    args = parser.parse_args()
    with args.summary.open() as handle:
        results = json.load(handle)

    count_pareto_front_methods(results)
    print("\n" + "=" * 80)
    analyze_network_contributions(results)


if __name__ == "__main__":
    main()
