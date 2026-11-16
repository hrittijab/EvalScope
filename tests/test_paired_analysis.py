
"""Tests for paired benchmark comparisons."""

import json

import pytest

from evalscope.data_loader import load_experiment
from evalscope.paired_analysis import (
    compare_defenses,
    exact_mcnemar_pvalue,
)


def make_dataset(tmp_path):
    records = []

    for i in range(20):
        for mode in ("baseline", "defended"):
            records.append({
                "attack_id": f"GEN-{i:03d}",
                "source_attack_id": f"SEED-{i // 2}",
                "category": "workflow_hijacking",
                "mode": mode,
                "attack_success": (
                    i < 12 if mode == "baseline" else i < 2
                ),
                "status": "completed",
                "valid_trial": True,
                "experiment_id": "test",
            })

    path = tmp_path / "paired.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return load_experiment(path)


def test_exact_mcnemar():
    assert exact_mcnemar_pvalue(0, 0) == 1.0
    assert exact_mcnemar_pvalue(1, 0) == 1.0
    assert exact_mcnemar_pvalue(10, 0) == pytest.approx(0.001953125)


def test_paired_improvement(tmp_path):
    df = make_dataset(tmp_path)

    result = compare_defenses(df)

    assert result["n_pairs"] == 20
    assert result["rate_a"] == pytest.approx(0.6)
    assert result["rate_b"] == pytest.approx(0.1)
    assert result["improvement"] == pytest.approx(0.5)
    assert result["a_only_success"] == 10
    assert result["b_only_success"] == 0


def test_cluster_comparison(tmp_path):
    df = make_dataset(tmp_path)

    result = compare_defenses(df, cluster=True)

    assert result["method"] == "cluster_bootstrap"
    assert result["n_clusters"] == 10
    assert result["ci_lower"] <= result["improvement"] <= result["ci_upper"]


def test_reproducibility(tmp_path):
    df = make_dataset(tmp_path)

    a = compare_defenses(df, seed=42)
    b = compare_defenses(df, seed=42)

    assert a == b


def test_no_pairs(tmp_path):
    df = make_dataset(tmp_path)
    df = df[df["mode"] == "baseline"]

    with pytest.raises(ValueError, match="No fully completed"):
        compare_defenses(df)
