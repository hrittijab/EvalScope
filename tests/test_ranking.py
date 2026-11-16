
"""Tests for benchmark ranking stability."""

import json

import pytest

from evalscope.data_loader import load_experiment
from evalscope.ranking import analyze_ranking_stability


def make_dataset(tmp_path):
    records = []

    for i in range(20):
        for mode in ("baseline", "prompt_guard", "authorization"):
            if mode == "baseline":
                success = i < 15
            elif mode == "prompt_guard":
                success = i < 8
            else:
                success = False

            records.append({
                "attack_id": f"GEN-{i:03d}",
                "source_attack_id": f"SEED-{i // 2}",
                "category": "workflow_hijacking",
                "mode": mode,
                "attack_success": success,
                "status": "completed",
                "valid_trial": True,
                "experiment_id": "test",
            })

    path = tmp_path / "ranking.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    return load_experiment(path)


def test_three_mode_ranking(tmp_path):
    df = make_dataset(tmp_path)

    result = analyze_ranking_stability(df)

    assert result["n_common_attacks"] == 20
    assert result["n_source_clusters"] == 10
    assert result["best_rank_probability"]["authorization"] > 0.99

def test_probabilities_sum_to_one(tmp_path):
    result = analyze_ranking_stability(make_dataset(tmp_path))

    assert sum(result["best_rank_probability"].values()) == pytest.approx(1.0)


def test_reproducibility(tmp_path):
    df = make_dataset(tmp_path)

    first = analyze_ranking_stability(df, seed=42)
    second = analyze_ranking_stability(df, seed=42)

    assert first == second


def test_two_modes(tmp_path):
    df = make_dataset(tmp_path)

    result = analyze_ranking_stability(
        df,
        modes=["baseline", "authorization"],
    )

    assert len(result["modes"]) == 2
    assert result["n_common_attacks"] == 20


def test_incomplete_exclusion(tmp_path):
    df = make_dataset(tmp_path)

    mask = (
        (df["attack_id"] == "GEN-001")
        & (df["mode"] == "prompt_guard")
    )

    df.loc[mask, "completed"] = False

    result = analyze_ranking_stability(df)

    assert result["n_common_attacks"] == 19
    assert result["n_excluded_attacks"] == 1


def test_requires_two_modes(tmp_path):
    df = make_dataset(tmp_path)

    with pytest.raises(ValueError):
        analyze_ranking_stability(df, modes=["baseline"])
