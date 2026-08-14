"""Prepare and publish the paper's canonical complete Pareto fronts.

Selection is derived from the checked-in complete-verification summaries. Each
experimental group is taken from its canonical paper budget: the main 10k
sample/1000s results for CIFAR-10 CNN7 and Tiny ImageNet, and the 1k
sample/300s results for MNIST and the additional architecture-study networks.
The publisher locates the resulting per-method Pareto checkpoints and builds a
checksummed Hugging Face model repository.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from .model_hub import PAPER_ROOT, sha256_file
except ImportError:  # Direct execution from the paper directory.
    from model_hub import PAPER_ROOT, sha256_file


GROUPING_KEYS = ("dataset", "architecture", "eps", "cert_train_method")
OBJECTIVE_KEYS = ("certified_accuracy", "clean_classification_accuracy")


@dataclass(frozen=True)
class SummarySpec:
    collection: str
    path: Path
    test_samples: int
    timeout_seconds: int
    group_filter: Callable[[Mapping[str, Any]], bool] | None = None


def _main_paper_group(result: Mapping[str, Any]) -> bool:
    return (
        result["architecture"] == "cnn7"
        and result["dataset"] in {"cifar10", "tinyimagenet"}
    )


def _short_budget_group(result: Mapping[str, Any]) -> bool:
    return result["dataset"] == "mnist" or (
        result["dataset"] == "cifar10" and result["architecture"] != "cnn7"
    )


DEFAULT_SUMMARIES = (
    SummarySpec(
        collection="main",
        path=PAPER_ROOT
        / "results"
        / "verification"
        / "main"
        / "summary_results.json",
        test_samples=10_000,
        timeout_seconds=1_000,
        group_filter=_main_paper_group,
    ),
    SummarySpec(
        collection="architecture_appendix",
        path=PAPER_ROOT
        / "results"
        / "verification"
        / "main"
        / "summary_results_timeout300_testsamples1000.json",
        test_samples=1_000,
        timeout_seconds=300,
        group_filter=_short_budget_group,
    ),
)

DATASET_METADATA = {
    "cifar10": {"input_shape": [3, 32, 32], "num_classes": 10},
    "mnist": {"input_shape": [1, 28, 28], "num_classes": 10},
    "tinyimagenet": {"input_shape": [3, 64, 64], "num_classes": 200},
}


def dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a >= b for a, b in zip(left, right)) and any(
        a > b for a, b in zip(left, right)
    )


def pareto_front(results: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        result
        for result in results
        if not any(
            other is not result
            and dominates(
                tuple(float(other[key]) for key in OBJECTIVE_KEYS),
                tuple(float(result[key]) for key in OBJECTIVE_KEYS),
            )
            for other in results
        )
    ]


def epsilon_label(epsilon: float) -> str:
    known = ((2 / 255, "2_255"), (8 / 255, "8_255"), (1 / 255, "1_255"))
    for value, label in known:
        if abs(float(epsilon) - value) < 1e-12:
            return label
    if abs(float(epsilon) - 0.3) < 1e-12:
        return "0_3"
    return format(float(epsilon), ".12g").replace(".", "_")


def training_epochs(epsilon: float) -> int:
    if abs(float(epsilon) - 8 / 255) < 1e-12:
        return 260
    if abs(float(epsilon) - 0.3) < 1e-12:
        return 70
    return 160


def _relative_summary_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PAPER_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def publication_records(
    summaries: Iterable[SummarySpec] = DEFAULT_SUMMARIES,
) -> list[dict[str, Any]]:
    """Return the union of final Pareto models used in paper artifacts."""
    records: dict[tuple[str, str, float, str, str], dict[str, Any]] = {}
    for summary in summaries:
        with summary.path.open(encoding="utf-8") as handle:
            results = json.load(handle)
        groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for result in results:
            if summary.group_filter is not None and not summary.group_filter(result):
                continue
            groups[tuple(result[key] for key in GROUPING_KEYS)].append(result)

        for group in groups.values():
            for result in pareto_front(group):
                epsilon = float(result["eps"])
                key = (
                    result["dataset"],
                    result["architecture"],
                    epsilon,
                    result["cert_train_method"],
                    result["hash"],
                )
                if result["dataset"] not in DATASET_METADATA:
                    raise ValueError(
                        f"No model metadata for dataset {result['dataset']!r}"
                    )
                label = epsilon_label(epsilon)
                method = result["cert_train_method"]
                model_id = "/".join(
                    [
                        result["dataset"],
                        result["architecture"],
                        f"eps-{label.replace('_', '-')}",
                        method,
                        result["hash"],
                    ]
                )
                checkpoint = "/".join(
                    [
                        "checkpoints",
                        result["dataset"],
                        result["architecture"],
                        f"eps_{label}",
                        method,
                        f"{result['hash']}.pt",
                    ]
                )
                record = records.setdefault(
                    key,
                    {
                        "model_id": model_id,
                        "config_hash": result["hash"],
                        "dataset": result["dataset"],
                        "architecture": result["architecture"],
                        "method": method,
                        "epsilon": epsilon,
                        "epsilon_label": label.replace("_", "/", 1)
                        if label.endswith("_255")
                        else label.replace("_", "."),
                        "training_epochs": training_epochs(epsilon),
                        **DATASET_METADATA[result["dataset"]],
                        "checkpoint": checkpoint,
                        "evaluations": [],
                    },
                )
                record["evaluations"].append(
                    {
                        "collection": summary.collection,
                        "summary": _relative_summary_path(summary.path),
                        "test_samples": summary.test_samples,
                        "timeout_seconds": summary.timeout_seconds,
                        "clean_accuracy": result[
                            "clean_classification_accuracy"
                        ],
                        "certified_accuracy": result["certified_accuracy"],
                        "adversarial_accuracy": result["adversarial_accuracy"],
                    }
                )

    return sorted(
        records.values(),
        key=lambda record: (
            record["dataset"],
            record["architecture"],
            record["epsilon"],
            record["method"],
            record["config_hash"],
        ),
    )


def checkpoint_index(
    roots: Iterable[str | Path], needed_hashes: set[str]
) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Checkpoint root does not exist: {root}")
        for checkpoint in root.rglob("*.pt"):
            if checkpoint.stem in needed_hashes:
                index[checkpoint.stem].append(checkpoint)
    return index


def resolve_checkpoints(
    records: Sequence[Mapping[str, Any]], roots: Iterable[str | Path]
) -> dict[str, Path]:
    needed = {record["config_hash"] for record in records}
    candidates = checkpoint_index(roots, needed)
    resolved = {}
    missing = []
    for config_hash in sorted(needed):
        paths = sorted(set(candidates.get(config_hash, [])))
        if not paths:
            missing.append(config_hash)
            continue
        if len(paths) > 1:
            digests = {sha256_file(path) for path in paths}
            if len(digests) != 1:
                locations = "\n  ".join(str(path) for path in paths)
                raise ValueError(
                    f"Conflicting checkpoints for {config_hash}:\n  {locations}"
                )
        resolved[config_hash] = min(
            paths, key=lambda path: (len(path.parts), str(path))
        )
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"Could not find {len(missing)} publication checkpoints: "
            f"{preview}{suffix}"
        )
    return resolved


def build_manifest(
    records: Sequence[Mapping[str, Any]],
    checkpoints: Mapping[str, Path],
    *,
    repo_id: str | None,
) -> dict[str, Any]:
    models = []
    for source_record in records:
        record = dict(source_record)
        checkpoint = checkpoints[record["config_hash"]]
        record["size_bytes"] = checkpoint.stat().st_size
        record["sha256"] = sha256_file(checkpoint)
        models.append(record)
    return {
        "schema_version": 1,
        "repo_id": repo_id,
        "paper": {
            "title": (
                "Rethinking Evaluation Paradigms in IBP-based Certified "
                "Training"
            ),
            "venue": "ICML 2026",
            "code": "https://github.com/ADA-research/CTRAIN",
        },
        "selection": (
            "Per-method nondominated fronts under each experiment's canonical "
            "complete-verification budget: 10k samples/1000s for CIFAR-10 "
            "CNN7 and Tiny ImageNet, and 1k samples/300s for MNIST and the "
            "additional architecture-study networks."
        ),
        "models": models,
    }


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def model_card(manifest: Mapping[str, Any]) -> str:
    repo_id = manifest.get("repo_id") or "YOUR_NAMESPACE/YOUR_REPOSITORY"
    return f"""---
library_name: CTRAIN
license: mit
tags:
- certified-robustness
- pareto-optimization
- pytorch
- icml-2026
---

# Pareto-front models for Rethinking Evaluation Paradigms

This repository contains {len(manifest['models'])} checkpoints on the final,
completely verified per-method Pareto fronts reported in **Rethinking
Evaluation Paradigms in IBP-based Certified Training** (ICML 2026). Each
dataset/network setting uses its canonical paper verification budget;
alternate-budget comparison fronts are not mixed into the model set.
`model_manifest.json` records each model's experiment metadata, reported
accuracies, byte size, and SHA-256 digest.

The checkpoints are CTRAIN bounded-model state dictionaries. Use the utility
in [ADA-research/CTRAIN](https://github.com/ADA-research/CTRAIN) to reconstruct
the correct architecture and wrapper:

```python
from papers.rethinking_evaluation_paradigms.model_hub import (
    list_models,
    load_model,
)

models = list_models(
    repo_id="{repo_id}",
    dataset="cifar10",
    architecture="cnn7",
    method="sabr",
    epsilon=2 / 255,
)
model = load_model(
    repo_id="{repo_id}",
    config_hash=models[0]["config_hash"],
    device="cuda",
)
logits = model(images)
```

Install CTRAIN and its Git-hosted bound-propagation dependencies before
loading a model:

```bash
pip install CTRAIN
ctrain-install-git-deps
pip install huggingface-hub
```

## Checkpoint organization

```text
checkpoints/<dataset>/<architecture>/<epsilon>/<method>/<config_hash>.pt
```

The weights are distributed under the CTRAIN repository's MIT license. Dataset
terms remain those of CIFAR-10, MNIST, and Tiny ImageNet.

## Citation

```bibtex
@inproceedings{{KauEtAl26,
  title = {{Rethinking Evaluation Paradigms in IBP-based Certified Training}},
  author = {{Kaulen, Konstantin and Shavit, Hadar and Hoos, Holger H}},
  booktitle = {{Proceedings of the 43rd International Conference on Machine
               Learning (ICML 2026)}},
  year = {{2026}}
}}
```
"""


def _ensure_staging_is_safe(staging_dir: Path, expected: set[Path]) -> None:
    if not staging_dir.exists():
        return
    allowed_cache_roots = {
        staging_dir / ".cache",
        staging_dir / ".cache" / "huggingface",
        staging_dir / ".cache" / ".huggingface",
    }
    for path in staging_dir.rglob("*"):
        if path in allowed_cache_roots or any(
            root in path.parents for root in allowed_cache_roots
        ):
            continue
        if path.is_file() and path not in expected:
            raise FileExistsError(
                f"Unexpected file in staging directory; refusing upload: {path}"
            )


def prepare_staging(
    staging_dir: str | Path,
    manifest: Mapping[str, Any],
    checkpoints: Mapping[str, Path],
) -> Path:
    """Create a resumable upload tree using hard links when possible."""
    staging_dir = Path(staging_dir).resolve()
    expected = {
        staging_dir / model["checkpoint"] for model in manifest["models"]
    }
    expected.update(
        {staging_dir / "model_manifest.json", staging_dir / "README.md"}
    )
    _ensure_staging_is_safe(staging_dir, expected)
    staging_dir.mkdir(parents=True, exist_ok=True)

    write_manifest(manifest, staging_dir / "model_manifest.json")
    (staging_dir / "README.md").write_text(
        model_card(manifest), encoding="utf-8"
    )
    for model in manifest["models"]:
        source = checkpoints[model["config_hash"]]
        target = staging_dir / model["checkpoint"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if (
                target.stat().st_size != model["size_bytes"]
                or sha256_file(target) != model["sha256"]
            ):
                raise FileExistsError(
                    f"Existing staging checkpoint does not match manifest: {target}"
                )
            continue
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    return staging_dir


def upload_repository(
    staging_dir: str | Path,
    *,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Publishing requires huggingface-hub; install the paper "
            "requirements first."
        ) from exc

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=staging_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and optionally upload the paper's final Pareto checkpoints."
        )
    )
    parser.add_argument(
        "--checkpoint-root",
        action="append",
        type=Path,
        required=True,
        help="Root to recursively search for <config_hash>.pt (repeatable).",
    )
    parser.add_argument("--repo-id")
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=PAPER_ROOT / "model_manifest.json",
    )
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.upload and not args.repo_id:
        raise ValueError("--repo-id is required with --upload")
    if args.upload and args.staging_dir is None:
        raise ValueError("--staging-dir is required with --upload")

    records = publication_records()
    checkpoints = resolve_checkpoints(records, args.checkpoint_root)
    manifest = build_manifest(records, checkpoints, repo_id=args.repo_id)
    manifest_path = write_manifest(manifest, args.manifest_output)
    total_bytes = sum(model["size_bytes"] for model in manifest["models"])
    print(
        f"Selected {len(records)} checkpoints ({total_bytes / 2**30:.2f} GiB); "
        f"wrote {manifest_path}"
    )

    if args.staging_dir is not None:
        staging_dir = prepare_staging(args.staging_dir, manifest, checkpoints)
        print(f"Prepared {staging_dir}")
        if args.upload:
            upload_repository(
                staging_dir,
                repo_id=args.repo_id,
                private=args.private,
            )
            print(f"Published https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
