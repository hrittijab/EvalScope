
"""Tests for bootstrap confidence intervals."""

import json

import pytest

from evalscope.data_loader import load_experiment
from evalscope.statistics import bootstrap_success_rate


def make_dataset(tmp_path):
    records = []

    for i in range(20):
        records.append({
            "attack_id": f"GEN-{i:03d}",
            "source_attack_id": f"SEED-{i // 2}",
            "category": "workflow_hijacking",
            "mode": "baseline",
            "attack_success": i < 10,
            "status": "completed",
            "valid_trial": True,
            "experiment_id": "test",
        })

    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    return load_experiment(path)


def test_bootstrap_rate(tmp_path):
    df = make_dataset(tmp_path)

    result = bootstrap_success_rate(df, "baseline")

    assert result["success_rate"] == 0.5
    assert result["ci_lower"] <= 0.5 <= result["ci_upper"]
    assert result["n_completed"] == 20


def test_cluster_bootstrap(tmp_path):
    df = make_dataset(tmp_path)

    result = bootstrap_success_rate(
        df, "baseline", cluster=True
    )

    assert result["method"] == "cluster_bootstrap"
    assert result["n_clusters"] == 10
    assert 0 <= result["ci_lower"] <= result["ci_upper"] <= 1


def test_reproducible(tmp_path):
    df = make_dataset(tmp_path)

    a = bootstrap_success_rate(df, "baseline", seed=42)
    b = bootstrap_success_rate(df, "baseline", seed=42)

    assert a == b


def test_invalid_confidence(tmp_path):
    df = make_dataset(tmp_path)

    with pytest.raises(ValueError):
        bootstrap_success_rate(
            df, "baseline", confidence=1.5
        )


def test_unknown_mode(tmp_path):
    df = make_dataset(tmp_path)

    with pytest.raises(ValueError):
        bootstrap_success_rate(df, "nonexistent")
