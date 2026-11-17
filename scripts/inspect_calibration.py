
"""Generate calibration diagnostics for AgentSecBench."""

from pathlib import Path

import matplotlib.pyplot as plt

from evalscope.data_loader import load_experiment
from evalscope.model_diagnostics import (
    plot_calibration,
    plot_probability_distributions,
)
from evalscope.predictive_model import evaluate_predictive_models


def main():
    df = load_experiment(
        "data/agentsecbench_experiment.json"
    )

    result = evaluate_predictive_models(
        df,
        n_splits=5,
        seed=42,
    )

    output = Path("results/agentsecbench")
    output.mkdir(parents=True, exist_ok=True)

    calibration = plot_calibration(
        result["predictions"],
        n_bins=8,
    )

    calibration.savefig(
        output / "model_calibration.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(calibration)

    distributions = plot_probability_distributions(
        result["predictions"]
    )

    distributions.savefig(
        output / "prediction_distributions.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(distributions)

    print("Calibration diagnostics generated successfully!")
    print(output / "model_calibration.png")
    print(output / "prediction_distributions.png")


if __name__ == "__main__":
    main()
