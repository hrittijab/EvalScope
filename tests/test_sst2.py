
"""Tests for SST-2 reliability and selective prediction."""

import numpy as np
import pandas as pd
import pytest

from evalscope.sst2_calibration import calibration_analysis
from evalscope.sst2_selective import selective_prediction_analysis


@pytest.fixture
def sample_predictions():
    return pd.DataFrame({
        "true_label": [1, 0, 1, 0],
        "predicted_label": [1, 0, 0, 1],
        "positive_probability": [0.95, 0.10, 0.20, 0.90],
        "confidence": [0.95, 0.90, 0.80, 0.90],
    })


def test_selective_prediction_full_coverage(sample_predictions):
    result = selective_prediction_analysis(sample_predictions)

    row = result.iloc[0]

    assert row["coverage"] == pytest.approx(1.0)
    assert row["risk"] == pytest.approx(0.5)


def test_selective_prediction_high_threshold(sample_predictions):
    result = selective_prediction_analysis(sample_predictions)

    row = result.iloc[
        (result["threshold"] - 0.95).abs().argmin()
    ]

    assert row["accepted"] == 1
    assert row["coverage"] == pytest.approx(0.25)
    assert row["risk"] == pytest.approx(0.0)


def test_selective_prediction_empty_acceptance(sample_predictions):
    result = selective_prediction_analysis(sample_predictions)

    row = result.iloc[-1]

    assert row["accepted"] == 0
    assert np.isnan(row["risk"])


def test_calibration_metrics(sample_predictions):
    diagnostics, fig = calibration_analysis(sample_predictions)

    try:
        assert 0 <= diagnostics["ece"] <= 1
        assert diagnostics["observed_accuracy"] == pytest.approx(0.5)
        assert diagnostics["high_confidence_errors"] == 0
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_calibration_perfect_predictions():
    df = pd.DataFrame({
        "true_label": [1, 0],
        "predicted_label": [1, 0],
        "confidence": [1.0, 1.0],
    })

    diagnostics, fig = calibration_analysis(df)

    try:
        assert diagnostics["ece"] == pytest.approx(0.0)
        assert diagnostics["observed_accuracy"] == pytest.approx(1.0)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
