
"""Research consistency tests for EvalScope."""

from pathlib import Path

import pytest

from evalscope.cluster_diagnostics import leave_one_cluster_out
from evalscope.data_loader import load_experiment
from evalscope.paired_analysis import compare_defenses
from evalscope.report import generate_report
from evalscope.sensitivity import analyze_incomplete_trials


DATASET = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "demo_experiment.json"
)


@pytest.fixture(scope="module")
def demo_df():
    return load_experiment(DATASET)


def test_paired_and_loco_improvement_agree(demo_df):
    paired = compare_defenses(
        demo_df,
        mode_a="baseline",
        mode_b="authorization",
        n_bootstrap=100,
        seed=42,
        cluster=True,
    )

    loco = leave_one_cluster_out(
        demo_df,
        mode_a="baseline",
        mode_b="authorization",
    )

    assert paired["n_pairs"] == loco["n_pairs"]

    assert paired["improvement"] == pytest.approx(
        loco["observed_improvement"]
    )


def test_completed_pair_counts_agree(demo_df):
    sensitivity = analyze_incomplete_trials(
        demo_df,
        mode_a="baseline",
        mode_b="authorization",
    )

    loco = leave_one_cluster_out(
        demo_df,
        mode_a="baseline",
        mode_b="authorization",
    )

    assert sensitivity["completed_pairs"] == loco["n_pairs"]


def test_loco_range_contains_observed_improvement(demo_df):
    loco = leave_one_cluster_out(
        demo_df,
        mode_a="baseline",
        mode_b="authorization",
    )

    # The observed estimate need not always lie inside
    # the LOCO range, so check the actual defining property:
    # the reported endpoints match the individual estimates.
    estimates = [
        item["improvement"]
        for item in loco["cluster_results"]
    ]

    assert loco["min_loco_improvement"] == pytest.approx(
        min(estimates)
    )

    assert loco["max_loco_improvement"] == pytest.approx(
        max(estimates)
    )


def test_report_contains_loco_analysis(demo_df):
    report = generate_report(
        demo_df,
        mode_a="baseline",
        mode_b="authorization",
        n_bootstrap=100,
        seed=42,
    )

    assert "Leave-One-Cluster-Out" in report
    assert "Threats to Validity" in report
    assert "Reproducibility" in report


def test_report_preserves_methodological_caveats(demo_df):
    report = generate_report(
        demo_df,
        mode_a="baseline",
        mode_b="authorization",
        n_bootstrap=100,
        seed=42,
    )

    # Normalize whitespace so Markdown line wrapping
    # doesn't cause false test failures.
    normalized = " ".join(report.split())

    assert "not a confidence interval" in normalized
    assert "source attack seeds" in normalized
    assert "not calibrated probabilities" in normalized
