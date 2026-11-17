
"""Tests for ML calibration visualizations."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from evalscope.model_diagnostics import (
    plot_calibration,
    plot_probability_distributions,
)


@pytest.fixture
def sample_predictions():
    return pd.DataFrame({
        "model": [
            "Constant Baseline",
            "Constant Baseline",
            "Logistic Regression",
            "Logistic Regression",
            "Random Forest",
            "Random Forest",
        ],
        "actual": [0, 1, 0, 1, 0, 1],
        "predicted_probability": [
            0.4, 0.4,
            0.1, 0.8,
            0.05, 0.95,
        ],
    })


def test_calibration_returns_figure(sample_predictions):
    figure = plot_calibration(
        sample_predictions,
        n_bins=5,
    )

    assert isinstance(figure, plt.Figure)
    assert len(figure.axes) == 1

    plt.close(figure)


def test_probability_distribution_returns_figure(sample_predictions):
    figure = plot_probability_distributions(
        sample_predictions
    )

    assert isinstance(figure, plt.Figure)
    assert len(figure.axes) == 3

    plt.close(figure)


def test_invalid_calibration_bins(sample_predictions):
    with pytest.raises(ValueError, match="at least 2"):
        plot_calibration(
            sample_predictions,
            n_bins=1,
        )
