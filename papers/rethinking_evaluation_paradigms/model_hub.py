"""Download and load the Pareto-optimal models reported in the paper.

The model repository contains the original CTRAIN ``state_dict`` checkpoints
and a manifest describing each model.  This module deliberately imports
PyTorch, CTRAIN's model wrappers, and ``huggingface_hub`` only when they are
needed so that listing the published models remains lightweight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


PAPER_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = PAPER_ROOT / "model_manifest.json"
DEFAULT_REPO_ID = os.environ.get("CTRAIN_PAPER_HF_REPO")
MANIFEST_FILENAME = "model_manifest.json"


def _read_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError(
            "Unsupported model manifest schema: "
            f"{manifest.get('schema_version')!r}"
        )
    models = manifest.get("models")
    if not isinstance(models, list):
        raise ValueError("Model manifest must contain a models list")

    required = {
        "model_id",
        "config_hash",
        "dataset",
        "architecture",
        "method",
        "epsilon",
        "checkpoint",
        "sha256",
    }
    ids = set()
    for model in models:
        missing = required.difference(model)
        if missing:
            raise ValueError(
                f"Manifest model is missing fields: {sorted(missing)}"
            )
        model_id = model["model_id"]
        if model_id in ids:
            raise ValueError(f"Duplicate model_id in manifest: {model_id}")
        ids.add(model_id)


def _import_hf_hub_download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Downloading paper checkpoints requires huggingface-hub. "
            "Install papers/rethinking_evaluation_paradigms/"
            "requirements-paper.txt."
        ) from exc
    return hf_hub_download


def load_manifest(
    *,
    manifest_path: str | Path | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: bool | str | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Load a local manifest or fetch the manifest from Hugging Face.

    An explicit ``repo_id`` selects the remote manifest.  Without one, the
    checked-in manifest is used, which makes model discovery work offline.
    ``CTRAIN_PAPER_HF_REPO`` can provide the default repository ID.
    """
    if manifest_path is not None:
        return _read_manifest(manifest_path)

    selected_repo = repo_id or DEFAULT_REPO_ID
    if selected_repo:
        hf_hub_download = _import_hf_hub_download()
        downloaded = hf_hub_download(
            repo_id=selected_repo,
            filename=MANIFEST_FILENAME,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            library_name="CTRAIN",
        )
        return _read_manifest(downloaded)

    if not DEFAULT_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Bundled manifest not found: {DEFAULT_MANIFEST_PATH}. "
            "Pass repo_id or manifest_path explicitly."
        )
    return _read_manifest(DEFAULT_MANIFEST_PATH)


def _model_matches(
    model: Mapping[str, Any],
    *,
    dataset: str | None,
    architecture: str | None,
    method: str | None,
    epsilon: float | None,
    collection: str | None,
    config_hash: str | None,
) -> bool:
    exact_filters = {
        "dataset": dataset,
        "architecture": architecture,
        "method": method,
        "config_hash": config_hash,
    }
    if any(
        value is not None and model.get(name) != value
        for name, value in exact_filters.items()
    ):
        return False
    if epsilon is not None and not math.isclose(
        float(model["epsilon"]), float(epsilon), rel_tol=0.0, abs_tol=1e-12
    ):
        return False
    if collection is not None and not any(
        evaluation.get("collection") == collection
        for evaluation in model.get("evaluations", [])
    ):
        return False
    return True


def list_models(
    *,
    manifest: Mapping[str, Any] | None = None,
    dataset: str | None = None,
    architecture: str | None = None,
    method: str | None = None,
    epsilon: float | None = None,
    collection: str | None = None,
    config_hash: str | None = None,
    **manifest_kwargs: Any,
) -> list[dict[str, Any]]:
    """List manifest entries, optionally restricted by experiment metadata."""
    if manifest is None:
        manifest = load_manifest(**manifest_kwargs)
    _validate_manifest(manifest)
    selected = [
        dict(model)
        for model in manifest["models"]
        if _model_matches(
            model,
            dataset=dataset,
            architecture=architecture,
            method=method,
            epsilon=epsilon,
            collection=collection,
            config_hash=config_hash,
        )
    ]
    return sorted(
        selected,
        key=lambda model: (
            model["dataset"],
            model["architecture"],
            float(model["epsilon"]),
            model["method"],
            model["config_hash"],
        ),
    )


def resolve_model(
    *,
    manifest: Mapping[str, Any] | None = None,
    model_id: str | None = None,
    config_hash: str | None = None,
    dataset: str | None = None,
    architecture: str | None = None,
    method: str | None = None,
    epsilon: float | None = None,
    collection: str | None = None,
    **manifest_kwargs: Any,
) -> dict[str, Any]:
    """Resolve filters to exactly one published model."""
    if manifest is None:
        manifest = load_manifest(**manifest_kwargs)
    if model_id is not None:
        matches = [
            dict(model)
            for model in manifest["models"]
            if model["model_id"] == model_id
        ]
    else:
        matches = list_models(
            manifest=manifest,
            config_hash=config_hash,
            dataset=dataset,
            architecture=architecture,
            method=method,
            epsilon=epsilon,
            collection=collection,
        )

    if not matches:
        raise LookupError("No published paper model matches the selection")
    if len(matches) > 1:
        examples = ", ".join(model["model_id"] for model in matches[:5])
        suffix = " ..." if len(matches) > 5 else ""
        raise ValueError(
            f"Selection matches {len(matches)} models; add more filters. "
            f"Matches include: {examples}{suffix}"
        )
    return matches[0]


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(
    *,
    manifest: Mapping[str, Any] | None = None,
    model_id: str | None = None,
    config_hash: str | None = None,
    dataset: str | None = None,
    architecture: str | None = None,
    method: str | None = None,
    epsilon: float | None = None,
    collection: str | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    token: bool | str | None = None,
    local_files_only: bool = False,
    verify: bool = True,
) -> Path:
    """Download one checkpoint and verify it against the paper manifest."""
    selected_repo = repo_id or (
        manifest.get("repo_id") if manifest is not None else None
    ) or DEFAULT_REPO_ID
    if manifest is None:
        manifest = load_manifest(
            repo_id=selected_repo,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        selected_repo = selected_repo or manifest.get("repo_id")
    if not selected_repo:
        raise ValueError(
            "No Hugging Face repository configured. Pass repo_id or set "
            "CTRAIN_PAPER_HF_REPO."
        )

    model = resolve_model(
        manifest=manifest,
        model_id=model_id,
        config_hash=config_hash,
        dataset=dataset,
        architecture=architecture,
        method=method,
        epsilon=epsilon,
        collection=collection,
    )
    hf_hub_download = _import_hf_hub_download()
    checkpoint = Path(
        hf_hub_download(
            repo_id=selected_repo,
            filename=model["checkpoint"],
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
            token=token,
            local_files_only=local_files_only,
            library_name="CTRAIN",
        )
    )
    if verify:
        actual_digest = sha256_file(checkpoint)
        if actual_digest != model["sha256"]:
            raise OSError(
                f"Checksum mismatch for {checkpoint}: expected "
                f"{model['sha256']}, got {actual_digest}"
            )
    return checkpoint


def download_models(
    *,
    manifest: Mapping[str, Any] | None = None,
    dataset: str | None = None,
    architecture: str | None = None,
    method: str | None = None,
    epsilon: float | None = None,
    collection: str | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    token: bool | str | None = None,
    local_files_only: bool = False,
    verify: bool = True,
) -> list[Path]:
    """Download every checkpoint matching an experiment selection."""
    if manifest is None:
        manifest = load_manifest(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    models = list_models(
        manifest=manifest,
        dataset=dataset,
        architecture=architecture,
        method=method,
        epsilon=epsilon,
        collection=collection,
    )
    if not models:
        raise LookupError("No published paper models match the selection")
    return [
        download_checkpoint(
            manifest=manifest,
            model_id=model["model_id"],
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
            token=token,
            local_files_only=local_files_only,
            verify=verify,
        )
        for model in models
    ]


def load_model(
    *,
    manifest: Mapping[str, Any] | None = None,
    model_id: str | None = None,
    config_hash: str | None = None,
    dataset: str | None = None,
    architecture: str | None = None,
    method: str | None = None,
    epsilon: float | None = None,
    collection: str | None = None,
    device: str = "cpu",
    eval_mode: bool = True,
    return_metadata: bool = False,
    **download_kwargs: Any,
):
    """Download, reconstruct, and load one CTRAIN paper model.

    The returned object is the original CTRAIN wrapper, so it supports normal
    forward passes as well as CTRAIN's incomplete and complete verification
    methods.  Building it requires the Git-hosted dependencies installed by
    ``ctrain-install-git-deps``.
    """
    if manifest is None:
        manifest = load_manifest(
            repo_id=download_kwargs.get("repo_id"),
            revision=download_kwargs.get("revision"),
            cache_dir=download_kwargs.get("cache_dir"),
            token=download_kwargs.get("token"),
            local_files_only=download_kwargs.get("local_files_only", False),
        )
    metadata = resolve_model(
        manifest=manifest,
        model_id=model_id,
        config_hash=config_hash,
        dataset=dataset,
        architecture=architecture,
        method=method,
        epsilon=epsilon,
        collection=collection,
    )
    checkpoint = download_checkpoint(
        manifest=manifest,
        model_id=metadata["model_id"],
        **download_kwargs,
    )

    try:
        import torch
        from papers.rethinking_evaluation_paradigms.mo_hpo.run_hpo import (
            build_model,
            build_wrapper,
        )
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Loading paper models requires CTRAIN and its Git-hosted "
            "dependencies. Install CTRAIN, then run ctrain-install-git-deps."
        ) from exc

    torch_device = torch.device(device)
    network = build_model(
        metadata["architecture"],
        metadata["input_shape"],
        metadata["num_classes"],
    )
    wrapper = build_wrapper(
        metadata["method"],
        model=network,
        input_shape=metadata["input_shape"],
        eps=float(metadata["epsilon"]),
        epochs=int(metadata["training_epochs"]),
        device=torch_device,
    )
    state_dict = torch.load(
        checkpoint,
        map_location=torch_device,
        weights_only=True,
    )
    wrapper.load_state_dict(state_dict)
    if eval_mode:
        wrapper.eval()
    if return_metadata:
        return wrapper, metadata
    return wrapper


def _format_models(models: Iterable[Mapping[str, Any]]) -> str:
    rows = []
    for model in models:
        rows.append(
            "\t".join(
                [
                    model["config_hash"],
                    model["dataset"],
                    model["architecture"],
                    model["method"],
                    model["epsilon_label"],
                ]
            )
        )
    header = "config_hash\tdataset\tarchitecture\tmethod\tepsilon"
    return "\n".join([header, *rows])


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id")
    parser.add_argument("--config-hash")
    parser.add_argument("--dataset")
    parser.add_argument("--architecture")
    parser.add_argument("--method")
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--collection")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover and download the paper's Pareto-front models."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision")
    parser.add_argument("--manifest", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List matching models.")
    _add_selection_arguments(list_parser)
    list_parser.add_argument("--json", action="store_true")

    download_parser = subparsers.add_parser(
        "download", help="Download one uniquely selected checkpoint."
    )
    _add_selection_arguments(download_parser)
    download_parser.add_argument("--local-dir", type=Path)
    download_parser.add_argument("--cache-dir", type=Path)
    download_parser.add_argument(
        "--all",
        action="store_true",
        help="Download every matching model instead of requiring one match.",
    )
    download_parser.add_argument("--no-verify", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(
        manifest_path=args.manifest,
        repo_id=args.repo_id if args.manifest is None else None,
        revision=args.revision,
    )
    selection = {
        "model_id": args.model_id,
        "config_hash": args.config_hash,
        "dataset": args.dataset,
        "architecture": args.architecture,
        "method": args.method,
        "epsilon": args.epsilon,
        "collection": args.collection,
    }
    if args.command == "list":
        models = list_models(manifest=manifest, **selection)
        if args.json:
            print(json.dumps(models, indent=2))
        else:
            print(_format_models(models))
        return 0

    if args.all:
        paths = download_models(
            manifest=manifest,
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=args.local_dir,
            cache_dir=args.cache_dir,
            verify=not args.no_verify,
            dataset=args.dataset,
            architecture=args.architecture,
            method=args.method,
            epsilon=args.epsilon,
            collection=args.collection,
        )
        print("\n".join(map(str, paths)))
    else:
        checkpoint = download_checkpoint(
            manifest=manifest,
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=args.local_dir,
            cache_dir=args.cache_dir,
            verify=not args.no_verify,
            **selection,
        )
        print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
