
"""Reproduce EvalScope statistical evaluation outputs."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Support running directly as:
# python scripts/reproduce.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evalscope.data_loader import load_experiment
from evalscope.report import generate_report, save_report
from evalscope.visualize import plot_evaluation_summary
from evalscope.ranking import analyze_ranking_stability
from scripts.generate_demo import generate_demo


def reproduce(
    dataset=None,
    output_dir="results/demo",
    n_bootstrap=2000,
    seed=42,
    mode_a="baseline",
    mode_b="authorization",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        dataset = ROOT / "data" / "demo_experiment.json"
        generate_demo(dataset, seed=42)

    dataset = Path(dataset).resolve()

    if not dataset.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset}"
        )

    if mode_a == mode_b:
        raise ValueError("Configurations must differ.")

    df = load_experiment(dataset)

    available_modes = set(df["mode"].unique())

    for mode in (mode_a, mode_b):
        if mode not in available_modes:
            raise ValueError(
                f"Configuration '{mode}' not found. "
                f"Available: {sorted(available_modes)}"
            )

    print("Dataset:", dataset)
    print("Records:", len(df))
    print("Unique attacks:", df["attack_id"].nunique())

    print("\nGenerating statistical figures...")

    figures = plot_evaluation_summary(
        df,
        output_dir=output_dir,
        mode_a=mode_a,
        mode_b=mode_b,
    )

    print("Generating research report...")

    report = generate_report(
        df,
        mode_a=mode_a,
        mode_b=mode_b,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    report_path = save_report(
        report,
        output_dir / "evaluation_report.md",
    )

    print("Analyzing ranking stability...")

    ranking = analyze_ranking_stability(
        df,
        modes=[mode_a, mode_b],
        n_bootstrap=n_bootstrap,
        seed=seed,
        cluster=True,
    )

    dataset_hash = hashlib.sha256(
        dataset.read_bytes()
    ).hexdigest()

    manifest = {
        "project": "EvalScope",
        "dataset": dataset.name,
        "dataset_sha256": dataset_hash,
        "experiment_ids": sorted(
            df["experiment_id"].dropna().unique().tolist()
        ),
        "records": len(df),
        "unique_attacks": int(df["attack_id"].nunique()),
        "source_clusters": int(
            df["source_attack_id"].nunique()
        ),
        "reference_configuration": mode_a,
        "comparison_configuration": mode_b,
        "bootstrap_resamples": n_bootstrap,
        "random_seed": seed,
        "ranking": ranking,
        "figures": [path.name for path in figures],
        "report": report_path.name,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    manifest_path = output_dir / "manifest.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nReproduction completed successfully!")

    for path in figures:
        print("Figure:", path)

    print("Report:", report_path)
    print("Manifest:", manifest_path)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce EvalScope statistical analysis"
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to benchmark JSON dataset",
    )

    parser.add_argument(
        "--output",
        default="results/demo",
        help="Output directory",
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap resamples",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Statistical random seed",
    )

    parser.add_argument(
        "--reference",
        default="baseline",
    )

    parser.add_argument(
        "--comparison",
        default="authorization",
    )

    args = parser.parse_args()

    reproduce(
        dataset=args.dataset,
        output_dir=args.output,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        mode_a=args.reference,
        mode_b=args.comparison,
    )
