
"""Regression tests for the six-defense evaluation."""

import pytest

from evalscope.six_defense_loader import (
    load_six_defense_attacks,
    load_six_defense_benign,
    security_utility_summary,
)
from evalscope.six_defense_statistics import (
    paired_comparison,
    add_holm_correction,
)


@pytest.fixture(scope="module")
def benchmark_data():
    attacks = load_six_defense_attacks(
        "data/six_defense_comparison.json"
    )
    benign = load_six_defense_benign(
        "data/benign_six_defense_comparison.json"
    )
    return attacks, benign


def test_record_counts(benchmark_data):
    attacks, benign = benchmark_data

    assert len(attacks) == 180
    assert len(benign) == 300
    assert attacks["completed"].all()


def test_security_utility_rates(benchmark_data):
    attacks, benign = benchmark_data

    summary = security_utility_summary(
        attacks, benign
    ).set_index("defense")

    assert summary.loc[
        "baseline", "attack_success_rate"
    ] == pytest.approx(7 / 30)

    assert summary.loc[
        "authorization", "attack_success_rate"
    ] == 0

    assert summary.loc[
        "authorization", "benign_success_rate"
    ] == 1

    assert summary.loc[
        "ml_layered", "benign_success_rate"
    ] == pytest.approx(48 / 50)


def test_paired_attack_comparisons(benchmark_data):
    attacks, _ = benchmark_data

    results = paired_comparison(
        attacks, n_bootstrap=1000
    ).set_index("defense")

    assert (results["matched_cases"] == 30).all()

    assert results.loc[
        "authorization", "difference"
    ] == pytest.approx(7 / 30)

    assert results.loc[
        "prompt_guard", "difference"
    ] == pytest.approx(6 / 30)


def test_holm_correction(benchmark_data):
    attacks, _ = benchmark_data

    results = add_holm_correction(
        paired_comparison(attacks, n_bootstrap=1000)
    )

    assert (results["holm_p"] >= results["mcnemar_p"]).all()
    assert not results["significant_05"].any()
    assert results["holm_p"].iloc[0] == pytest.approx(
        0.078125
    )


def test_benign_comparisons(benchmark_data):
    _, benign = benchmark_data

    results = paired_comparison(
        benign, n_bootstrap=1000
    ).set_index("defense")

    assert (results["matched_cases"] == 50).all()

    assert results.loc[
        "ml_layered", "difference"
    ] == pytest.approx(2 / 50)
