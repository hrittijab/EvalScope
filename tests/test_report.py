
"""Tests for automated Markdown report generation."""

import json

import pytest

from evalscope.data_loader import load_experiment
from evalscope.report import generate_report, save_report


@pytest.fixture
def sample_data(tmp_path):
    records = []

    for i in range(12):
        for mode in ("baseline", "defended"):
            records.append({
                "attack_id": f"ATTACK-{i:03d}",
                "source_attack_id": f"SEED-{i // 3}",
                "category": "instruction_override",
                "mode": mode,
                "attack_success": (
                    i < 7 if mode == "baseline" else False
                ),
                "status": "completed",
                "valid_trial": True,
                "experiment_id": "test-experiment",
            })

    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    return load_experiment(path)


def test_report_contains_sections(sample_data):
    report = generate_report(
        sample_data,
        n_bootstrap=100,
    )

    assert "# EvalScope" in report
    assert "Executive Summary" in report
    assert "Attack Success Rates" in report
    assert "Paired Defense Comparison" in report
    assert "Incomplete-Trial Sensitivity" in report
    assert "Ranking Stability" in report
    assert "Threats to Validity" in report


def test_report_contains_metrics(sample_data):
    report = generate_report(
        sample_data,
        n_bootstrap=100,
    )

    assert "12" in report
    assert "baseline" in report
    assert "defended" in report
    assert "58.3%" in report


def test_report_saved(sample_data, tmp_path):
    report = generate_report(
        sample_data,
        n_bootstrap=100,
    )

    output = save_report(
        report,
        tmp_path / "reports" / "evaluation.md",
    )

    assert output.exists()
    assert output.read_text(encoding="utf-8") == report


def test_same_modes_rejected(sample_data):
    with pytest.raises(ValueError):
        generate_report(
            sample_data,
            mode_a="baseline",
            mode_b="baseline",
            n_bootstrap=100,
        )
