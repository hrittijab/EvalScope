
"""Tests for incomplete-trial sensitivity analysis."""

import json

import pytest

from evalscope.data_loader import load_experiment
from evalscope.sensitivity import analyze_incomplete_trials


def make_dataset(tmp_path):
    records = [
        ("A1", "baseline", True, "completed"),
        ("A1", "defended", False, "completed"),
        ("A2", "baseline", False, "truncated"),
        ("A2", "defended", False, "completed"),
        ("A3", "baseline", False, "completed"),
        ("A3", "defended", False, "timeout"),
    ]

    data = [
        {
            "attack_id": attack_id,
            "source_attack_id": "SEED-001",
            "category": "workflow_hijacking",
            "mode": mode,
            "attack_success": success,
            "status": status,
            "valid_trial": True,
            "experiment_id": "test",
        }
        for attack_id, mode, success, status in records
    ]

    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    return load_experiment(path)


def test_pair_counts(tmp_path):
    result = analyze_incomplete_trials(make_dataset(tmp_path))

    assert result["total_pairs"] == 3
    assert result["completed_pairs"] == 1
    assert result["incomplete_pairs"] == 2
    assert result["incomplete_a"] == 1
    assert result["incomplete_b"] == 1


def test_completed_improvement(tmp_path):
    result = analyze_incomplete_trials(make_dataset(tmp_path))

    assert result["completed_only_improvement"] == 1.0


def test_sensitivity_bounds(tmp_path):
    result = analyze_incomplete_trials(make_dataset(tmp_path))

    assert result["best_case_improvement"] == pytest.approx(2 / 3)
    assert result["worst_case_improvement"] == pytest.approx(0.0)


def test_no_pairs(tmp_path):
    df = make_dataset(tmp_path)
    df = df[df["mode"] == "baseline"]

    with pytest.raises(ValueError, match="No paired"):
        analyze_incomplete_trials(df)
