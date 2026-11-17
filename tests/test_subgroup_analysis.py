
"""Tests for subgroup effectiveness analysis."""

import numpy as np
import pandas as pd
import pytest

from evalscope.subgroup_analysis import analyze_subgroups


def make_dataset():
    """Create a small paired dataset with two categories."""

    rows = []

    attacks = [
        ("A1", "S1", "override", True, False),
        ("A2", "S1", "override", True, False),
        ("A3", "S2", "override", False, False),
        ("A4", "S3", "hijacking", True, False),
        ("A5", "S4", "hijacking", False, False),
        ("A6", "S5", "hijacking", True, True),
    ]

    for attack_id, source_id, category, baseline, defended in attacks:
        for mode, success in (
            ("baseline", baseline),
            ("defended", defended),
        ):
            rows.append({
                "attack_id": attack_id,
                "source_attack_id": source_id,
                "category": category,
                "mode": mode,
                "attack_success": success,
                "status": "completed",
                "valid_trial": True,
                "completed": True,
                "experiment_id": "subgroup-test",
            })

    return pd.DataFrame(rows)


def test_subgroup_counts():
    results = analyze_subgroups(
        make_dataset(),
        n_bootstrap=100,
        seed=42,
    )

    assert len(results) == 2
    assert results["n_pairs"].sum() == 6


def test_subgroup_improvement():
    results = analyze_subgroups(
        make_dataset(),
        n_bootstrap=100,
        seed=42,
    )

    override = results.set_index("category").loc["override"]
    hijacking = results.set_index("category").loc["hijacking"]

    assert override["improvement"] == pytest.approx(2 / 3)
    assert hijacking["improvement"] == pytest.approx(1 / 3)


def test_cluster_counts():
    results = analyze_subgroups(
        make_dataset(),
        n_bootstrap=100,
        seed=42,
    )

    by_category = results.set_index("category")

    assert by_category.loc["override", "n_clusters"] == 2
    assert by_category.loc["hijacking", "n_clusters"] == 3


def test_bootstrap_reproducibility():
    first = analyze_subgroups(
        make_dataset(),
        n_bootstrap=100,
        seed=42,
    )

    second = analyze_subgroups(
        make_dataset(),
        n_bootstrap=100,
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)


def test_missing_category_rejected():
    df = make_dataset().drop(columns=["category"])

    with pytest.raises(ValueError, match="Grouping column"):
        analyze_subgroups(df)


def test_same_modes_rejected():
    with pytest.raises(ValueError, match="different"):
        analyze_subgroups(
            make_dataset(),
            mode_a="baseline",
            mode_b="baseline",
        )


def test_small_subgroup_warning():
    results = analyze_subgroups(
        make_dataset(),
        n_bootstrap=100,
        seed=42,
        min_pairs=10,
    )

    assert results["warning"].str.contains(
        "Few completed pairs"
    ).all()


def test_invalid_bootstrap_count():
    with pytest.raises(ValueError, match="n_bootstrap"):
        analyze_subgroups(
            make_dataset(),
            n_bootstrap=0,
        )


def test_confidence_interval_bounds():
    results = analyze_subgroups(
        make_dataset(),
        n_bootstrap=100,
        seed=42,
    )

    for _, row in results.iterrows():
        assert np.isfinite(row["ci_lower"])
        assert np.isfinite(row["ci_upper"])
        assert -1 <= row["ci_lower"] <= 1
        assert -1 <= row["ci_upper"] <= 1
        assert row["ci_lower"] <= row["ci_upper"]
