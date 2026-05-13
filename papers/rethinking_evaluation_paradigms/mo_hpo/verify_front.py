import argparse
import csv
from pathlib import Path

import torch

from run_hpo import build_loaders, build_model, build_wrapper


def main():
    parser = argparse.ArgumentParser(description="Run complete verification for checkpoints listed in a front CSV.")
    parser.add_argument("--front", required=True, help="CSV produced by calculate_fronts.py.")
    parser.add_argument("--dataset", required=True, choices=["cifar10", "mnist", "gtsrb", "tinyimagenet"])
    parser.add_argument("--network", required=True, choices=["cnn3", "cnn5", "cnn7", "cnn9", "cnn11", "wide_cnn7", "narrow_cnn7", "resnet18"])
    parser.add_argument("--method", required=True, choices=["mtl_ibp", "sabr", "shi", "crown_ibp", "crown_ibp_nofusion"])
    parser.add_argument("--eps", required=True, type=float)
    parser.add_argument("--timeout", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=10_000)
    parser.add_argument("--abcrown-batch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--results-root", default="papers/rethinking_evaluation_paradigms/results/verification")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    (_, test_loader), input_shape, n_classes = build_loaders(
        args.dataset, args.network, args.batch_size, args.data_root
    )
    device = torch.device(args.device)

    with open(args.front, newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        config_hash = row["config_hash"]
        if not config_hash:
            raise ValueError("Front CSV does not contain config_hash values. Re-run calculate_fronts.py.")

        study_path = Path(row["source_study"]).resolve()
        checkpoint_path = study_path.parent / "nets" / f"{config_hash}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = build_model(args.network, input_shape, n_classes)
        wrapper = build_wrapper(
            args.method,
            model=model,
            input_shape=input_shape,
            eps=args.eps,
            epochs=1,
            device=device,
        )
        wrapper.load_state_dict(torch.load(checkpoint_path, map_location=device))
        wrapper.eval()

        results_path = (
            Path(args.results_root)
            / args.dataset
            / args.network
            / str(args.eps)
            / args.method
            / config_hash
        )
        print(f"Verifying {checkpoint_path}")
        print(
            wrapper.evaluate_complete(
                test_loader,
                test_samples=args.test_samples,
                timeout=args.timeout,
                abcrown_batch_size=args.abcrown_batch_size,
                results_path=str(results_path),
                warm_start=True,
            )
        )


if __name__ == "__main__":
    main()
