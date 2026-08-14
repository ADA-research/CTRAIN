import json
from pathlib import Path

import pytest

from papers.rethinking_evaluation_paradigms import model_hub


def sample_manifest():
    base = {
        "dataset": "cifar10",
        "architecture": "cnn7",
        "epsilon": 2 / 255,
        "epsilon_label": "2/255",
        "input_shape": [3, 32, 32],
        "num_classes": 10,
        "training_epochs": 160,
        "sha256": "0" * 64,
        "size_bytes": 1,
    }
    return {
        "schema_version": 1,
        "repo_id": "example/models",
        "models": [
            {
                **base,
                "model_id": "cifar10/cnn7/eps-2-255/sabr/a",
                "config_hash": "a",
                "method": "sabr",
                "checkpoint": "checkpoints/a.pt",
                "evaluations": [{"collection": "main"}],
            },
            {
                **base,
                "model_id": "cifar10/cnn7/eps-2-255/shi/b",
                "config_hash": "b",
                "method": "shi",
                "checkpoint": "checkpoints/b.pt",
                "evaluations": [{"collection": "architecture_appendix"}],
            },
        ],
    }


def test_load_local_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(sample_manifest()), encoding="utf-8")
    assert model_hub.load_manifest(manifest_path=path)["repo_id"] == (
        "example/models"
    )


def test_list_models_filters_metadata_and_collection():
    selected = model_hub.list_models(
        manifest=sample_manifest(),
        dataset="cifar10",
        architecture="cnn7",
        method="sabr",
        epsilon=2 / 255,
        collection="main",
    )
    assert [model["config_hash"] for model in selected] == ["a"]


def test_resolve_model_requires_unique_selection():
    with pytest.raises(ValueError, match="matches 2 models"):
        model_hub.resolve_model(manifest=sample_manifest(), dataset="cifar10")
    assert model_hub.resolve_model(
        manifest=sample_manifest(), config_hash="b"
    )["method"] == "shi"


def test_sha256_file(tmp_path):
    path = tmp_path / "weights.pt"
    path.write_bytes(b"paper checkpoint")
    assert model_hub.sha256_file(path) == (
        "1f0fb55538031a31a429f66246422883da84a7ba00deb750695b006dead0261b"
    )


def test_download_checkpoint_verifies_content(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"paper checkpoint")
    manifest = sample_manifest()
    manifest["models"][0]["sha256"] = model_hub.sha256_file(checkpoint)
    monkeypatch.setattr(
        model_hub, "_import_hf_hub_download", lambda: lambda **kwargs: checkpoint
    )

    downloaded = model_hub.download_checkpoint(
        manifest=manifest, config_hash="a"
    )

    assert downloaded == checkpoint


def test_download_models_downloads_every_match(tmp_path, monkeypatch):
    first = tmp_path / "a.pt"
    second = tmp_path / "b.pt"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    manifest = sample_manifest()
    paths = {"checkpoints/a.pt": first, "checkpoints/b.pt": second}
    for model in manifest["models"]:
        model["sha256"] = model_hub.sha256_file(paths[model["checkpoint"]])
    monkeypatch.setattr(
        model_hub,
        "_import_hf_hub_download",
        lambda: lambda **kwargs: paths[kwargs["filename"]],
    )

    downloaded = model_hub.download_models(
        manifest=manifest, dataset="cifar10"
    )

    assert downloaded == [first, second]


def test_bundled_manifest_is_valid():
    manifest = model_hub.load_manifest(
        manifest_path=Path(model_hub.DEFAULT_MANIFEST_PATH)
    )
    assert len(manifest["models"]) == 145
    assert sum(
        "main" in {item["collection"] for item in model["evaluations"]}
        for model in manifest["models"]
    ) == 69
