
"""Verify the core EvalScope research workflow."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from evalscope.data_loader import load_experiment
from evalscope.model_diagnostics import (
    plot_calibration,
    plot_probability_distributions,
)
from evalscope.predictive_model import evaluate_predictive_models
from evalscope.report import generate_report, save_report


def main():
    dataset = Path("data/agentsecbench_experiment.json")

    if not dataset.exists():
        dataset = Path("data/demo_experiment.json")

    if not dataset.exists():
        raise FileNotFoundError(
            "No experiment dataset found in data/."
        )

    output = Path("results/verification")
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {dataset}")
    df = load_experiment(dataset)

    print("Evaluating predictive models...")
    result = evaluate_predictive_models(
        df,
        n_splits=5,
        seed=42,
    )

    result["metrics"].to_csv(
        output / "model_metrics.csv",
        index=False,
    )

    result["predictions"].to_csv(
        output / "model_predictions.csv",
        index=False,
    )

    print(result["metrics"].round(4).to_string(index=False))

    print("Generating calibration diagnostics...")

    figures = {
        "model_calibration.png": plot_calibration(
            result["predictions"]
        ),
        "prediction_distributions.png":
            plot_probability_distributions(
                result["predictions"]
            ),
    }

    for filename, figure in figures.items():
        figure.savefig(
            output / filename,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)

    print("Generating research report...")

    report = generate_report(
        df,
        mode_a="baseline",
        mode_b="defended",
        n_bootstrap=2000,
        seed=42,
    )

    save_report(
        report,
        output / "research_report.md",
    )

    print("Verification completed successfully.")
    print(f"Artifacts saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
