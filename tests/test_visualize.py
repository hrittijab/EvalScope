
"""Tests for statistical visualization output."""

import json

from evalscope.data_loader import load_experiment
from evalscope.visualize import plot_evaluation_summary


def test_visualization_files(tmp_path):
    records = []

    for i in range(12):
        for mode in ("baseline", "defended"):
            records.append({
                "attack_id": f"GEN-{i:03d}",
                "source_attack_id": f"SEED-{i // 3}",
                "category": "workflow_hijacking",
                "mode": mode,
                "attack_success": (
                    i < 7 if mode == "baseline" else False
                ),
                "status": "completed",
                "valid_trial": True,
                "experiment_id": "test",
            })

    source = tmp_path / "experiment.json"
    source.write_text(json.dumps(records), encoding="utf-8")

    df = load_experiment(source)
    output_dir = tmp_path / "figures"

    paths = plot_evaluation_summary(df, output_dir)

    assert len(paths) == 3

    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 1000
        assert path.suffix == ".png"
