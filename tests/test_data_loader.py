
"""Tests for benchmark data loading and pairing."""

import json

import pytest

from evalscope.data_loader import (
    load_experiment,
    summarize_experiment,
    paired_trials,
)


def make_record(attack_id, mode, status="completed"):
    return {
        "attack_id": attack_id,
        "source_attack_id": "ATTACK-001",
        "category": "workflow_hijacking",
        "mode": mode,
        "attack_success": mode == "baseline",
        "status": status,
        "valid_trial": True,
        "experiment_id": "test-experiment",
    }


def save_records(tmp_path, records):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_load_experiment(tmp_path):
    path = save_records(
        tmp_path,
        [
            make_record("GEN-001", "baseline"),
            make_record("GEN-001", "defended"),
        ],
    )

    df = load_experiment(path)

    assert len(df) == 2
    assert df["attack_id"].nunique() == 1
    assert df["completed"].sum() == 2


def test_reject_missing_fields(tmp_path):
    record = make_record("GEN-001", "baseline")
    del record["attack_success"]

    path = save_records(tmp_path, [record])

    with pytest.raises(ValueError, match="missing fields"):
        load_experiment(path)


def test_reject_duplicate_trials(tmp_path):
    record = make_record("GEN-001", "baseline")
    path = save_records(tmp_path, [record, record])

    with pytest.raises(ValueError, match="Duplicate"):
        load_experiment(path)


def test_pair_completed_trials(tmp_path):
    path = save_records(
        tmp_path,
        [
            make_record("GEN-001", "baseline"),
            make_record("GEN-001", "defended"),
            make_record("GEN-002", "baseline"),
            make_record("GEN-002", "defended", "truncated"),
        ],
    )

    df = load_experiment(path)

    completed_pairs = paired_trials(df)
    all_pairs = paired_trials(df, completed_only=False)

    assert len(completed_pairs) == 1
    assert len(all_pairs) == 2


def test_summary(tmp_path):
    path = save_records(
        tmp_path,
        [
            make_record("GEN-001", "baseline"),
            make_record("GEN-001", "defended", "timeout"),
        ],
    )

    summary = summarize_experiment(load_experiment(path))

    assert summary["total_trials"] == 2
    assert summary["unique_attacks"] == 1
    assert summary["completed_trials"] == 1
    assert summary["incomplete_trials"] == 1
