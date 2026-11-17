
"""Tests for leave-one-cluster-out robustness analysis."""

import pandas as pd
import pytest

from evalscope.cluster_diagnostics import leave_one_cluster_out


def make_dataset():
    """Create a small paired benchmark with three source seeds."""
    records = []

    cases = [
        ("A1", "S1", True, False),
        ("A2", "S1", True, False),
        ("A3", "S2", False, False),
        ("A4", "S2", True, False),
        ("A5", "S3", True, False),
        ("A6", "S3", False, False),
    ]

    for attack_id, source_id, baseline, defended in cases:
        for mode, success in [
            ("baseline", baseline),
            ("defended", defended),
        ]:
            records.append({
                "attack_id": attack_id,
                "source_attack_id": source_id,
                "category": "synthetic",
                "mode": mode,
                "attack_success": success,
                "status": "completed",
                "valid_trial": True,
                "completed": True,
                "experiment_id": "test-experiment",
            })

    return pd.DataFrame(records)


def test_loco_returns_expected_structure():
    result = leave_one_cluster_out(make_dataset())

    assert result["n_pairs"] == 6
    assert result["n_clusters"] == 3
    assert len(result["cluster_results"]) == 3


def test_loco_observed_improvement():
    result = leave_one_cluster_out(make_dataset())

    assert result["observed_improvement"] == pytest.approx(
        4 / 6
    )


def test_loco_estimates():
    result = leave_one_cluster_out(make_dataset())

    # Removing S1 leaves 2 successes in 4 pairs.
    # Removing S2 leaves 3 successes in 4 pairs.
    # Removing S3 leaves 3 successes in 4 pairs.
    assert result["min_loco_improvement"] == pytest.approx(
        0.5
    )
    assert result["max_loco_improvement"] == pytest.approx(
        0.75
    )

    assert result["all_loco_positive"] is True


def test_loco_rejects_single_cluster():
    df = make_dataset()
    df = df[df["source_attack_id"] == "S1"]

    with pytest.raises(ValueError, match="At least two"):
        leave_one_cluster_out(df)


def test_loco_rejects_no_completed_pairs():
    df = make_dataset()
    df["completed"] = False

    with pytest.raises(ValueError, match="No completed"):
        leave_one_cluster_out(df)
