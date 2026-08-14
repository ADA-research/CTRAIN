import json

import pytest

from papers.rethinking_evaluation_paradigms.publish_models import (
    SummarySpec,
    epsilon_label,
    publication_records,
)


def result(config_hash, clean, certified):
    return {
        "dataset": "cifar10",
        "architecture": "cnn7",
        "eps": str(2 / 255),
        "cert_train_method": "sabr",
        "hash": config_hash,
        "clean_classification_accuracy": clean,
        "certified_accuracy": certified,
        "adversarial_accuracy": certified + 1,
    }


def test_publication_records_selects_pareto_union(tmp_path):
    first = tmp_path / "main.json"
    second = tmp_path / "appendix.json"
    first.write_text(
        json.dumps(
            [
                result("natural", 80, 50),
                result("robust", 70, 60),
                result("dominated", 60, 40),
            ]
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps([result("natural", 80, 51), result("extra", 75, 59)]),
        encoding="utf-8",
    )
    summaries = (
        SummarySpec("main", first, 10_000, 1_000),
        SummarySpec("architecture_appendix", second, 1_000, 300),
    )

    records = publication_records(summaries)

    assert {record["config_hash"] for record in records} == {
        "natural",
        "robust",
        "extra",
    }
    natural = next(
        record for record in records if record["config_hash"] == "natural"
    )
    assert {item["collection"] for item in natural["evaluations"]} == {
        "main",
        "architecture_appendix",
    }


def test_default_publication_records_use_canonical_budgets():
    records = publication_records()
    mtl_cifar10_cnn7 = {
        record["config_hash"]
        for record in records
        if record["dataset"] == "cifar10"
        and record["architecture"] == "cnn7"
        and record["method"] == "mtl_ibp"
        and record["epsilon"] == pytest.approx(2 / 255)
    }

    assert len(records) == 145
    assert mtl_cifar10_cnn7 == {
        "349591858b2fc0dd0c20ce7feb24a693",
        "8726ba3c7f6a6884b30d2d0025fb1337",
        "da5508650a01836b067e42738ad74650",
        "ee1d0628962269d5816719a980d9b574",
    }


@pytest.mark.parametrize(
    ("epsilon", "label"),
    [(2 / 255, "2_255"), (8 / 255, "8_255"), (1 / 255, "1_255"), (0.3, "0_3")],
)
def test_epsilon_label(epsilon, label):
    assert epsilon_label(epsilon) == label
