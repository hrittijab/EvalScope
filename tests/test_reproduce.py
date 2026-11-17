
"""Integration tests for the reproducibility pipeline."""

import json
from pathlib import Path

from scripts.reproduce import reproduce



def test_reproduce_generates_all_outputs(tmp_path):
    output_dir = tmp_path / "results"

    manifest = reproduce(
        dataset=None,
        output_dir=output_dir,
        n_bootstrap=100,
        seed=42,
    )

    assert manifest["records"] == 480
    assert manifest["unique_attacks"] == 120
    assert manifest["source_clusters"] == 30

    assert len(manifest["figures"]) == 3

    for filename in manifest["figures"]:
        path = output_dir / filename
        assert path.exists()
        assert path.stat().st_size > 1000

    report = output_dir / "evaluation_report.md"
    assert report.exists()
    assert "Executive Summary" in report.read_text(
        encoding="utf-8"
    )

    manifest_file = output_dir / "manifest.json"
    assert manifest_file.exists()

    saved = json.loads(
        manifest_file.read_text(encoding="utf-8")
    )

    assert saved["dataset_sha256"] == (
        manifest["dataset_sha256"]
    )


def test_reproduce_deterministic_results(tmp_path):
    first = reproduce(
        output_dir=tmp_path / "first",
        n_bootstrap=100,
        seed=42,
    )

    second = reproduce(
        output_dir=tmp_path / "second",
        n_bootstrap=100,
        seed=42,
    )

    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["ranking"] == second["ranking"]
