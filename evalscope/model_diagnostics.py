
"""Calibration and prediction diagnostics for EvalScope."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve


def plot_calibration(
    predictions: pd.DataFrame,
    n_bins: int = 8,
):
    """Plot out-of-fold reliability curves for all models."""

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Perfect calibration",
    )

    for model_name, group in predictions.groupby("model"):
        y_true = group["actual"].to_numpy(dtype=int)
        y_prob = group["predicted_probability"].to_numpy(
            dtype=float
        )

        observed, predicted = calibration_curve(
            y_true,
            y_prob,
            n_bins=n_bins,
            strategy="uniform",
        )

        ax.plot(
            predicted,
            observed,
            marker="o",
            label=model_name,
        )

    ax.set_xlabel("Mean Predicted Attack Success Probability")
    ax.set_ylabel("Observed Attack Success Frequency")
    ax.set_title("Out-of-Fold Model Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig


def plot_probability_distributions(
    predictions: pd.DataFrame,
):
    """Visualize prediction probabilities by actual outcome."""

    models = predictions["model"].unique()
    n_models = len(models)

    fig, axes = plt.subplots(
        n_models,
        1,
        figsize=(9, max(3, n_models * 2.5)),
        squeeze=False,
    )

    for index, model_name in enumerate(models):
        ax = axes[index, 0]

        group = predictions[
            predictions["model"] == model_name
        ]

        for outcome in (0, 1):
            subset = group[
                group["actual"] == outcome
            ]

            ax.hist(
                subset["predicted_probability"],
                bins=np.linspace(0, 1, 16),
                alpha=0.6,
                label=f"Actual outcome: {outcome}",
            )

        ax.set_title(model_name)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Number of Evaluations")
        ax.legend()

    fig.tight_layout()
    return fig
