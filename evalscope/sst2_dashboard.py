
"""Streamlit dashboard for SST-2 model reliability."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

from evalscope.sst2_calibration import calibration_analysis
from evalscope.sst2_selective import selective_prediction_analysis


@st.cache_data(show_spinner=False)
def load_sst2_results():
    """Load locally stored SST-2 predictions and calculate metrics."""

    path = Path(__file__).resolve().parent.parent / "data" / "sst2_predictions.csv"

    predictions = pd.read_csv(path)

    y_true = predictions["true_label"].to_numpy()
    y_pred = predictions["predicted_label"].to_numpy()
    probabilities = predictions["positive_probability"].to_numpy()

    correct = int((y_true == y_pred).sum())
    total = len(predictions)

    interval = binomtest(correct, total).proportion_ci(
        confidence_level=0.95,
        method="exact",
    )

    metrics = {
        "examples": total,
        "correct": correct,
        "accuracy": accuracy_score(y_true, y_pred),
        "ci_lower": interval.low,
        "ci_upper": interval.high,
        "brier_score": brier_score_loss(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ),
    }

    return predictions, metrics


def render_sst2_dashboard():
    """Render the SST-2 NLP reliability dashboard."""

    st.header("NLP Model Reliability — SST-2")

    st.caption(
        "Independent benchmark: 872 matched SST-2 validation "
        "predictions from a public Rubrix dataset."
    )

    try:
        predictions, metrics = load_sst2_results()

    except Exception as exc:
        st.warning(f"Unable to load SST-2 benchmark: {exc}")
        return

    diagnostics, fig = calibration_analysis(predictions)

    # --------------------------------------------------
    # 1. Overview Metrics
    # --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy']:.2%}",
    )

    col2.metric(
        "95% CI",
        f"{metrics['ci_lower']:.1%}–{metrics['ci_upper']:.1%}",
    )

    col3.metric(
        "Expected Calibration Error",
        f"{diagnostics['ece']:.2%}",
    )

    col4.metric(
        "High-Confidence Errors",
        diagnostics["high_confidence_errors"],
    )

    # --------------------------------------------------
    # 2. Confidence Calibration
    # --------------------------------------------------

    st.subheader("Confidence Calibration")

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        f"Average model confidence: "
        f"{diagnostics['mean_confidence']:.2%}. "
        f"Observed accuracy: "
        f"{diagnostics['observed_accuracy']:.2%}. "
        "ECE uses 10 equal-width confidence bins."
    )

    # --------------------------------------------------
    # 3. Selective Prediction
    # --------------------------------------------------

    st.subheader("Selective Prediction — Risk vs Coverage")

    st.write(
        "Evaluate whether abstaining from low-confidence "
        "predictions reduces classification errors."
    )

    selective = selective_prediction_analysis(predictions)

    fig_selective, ax = plt.subplots(figsize=(8, 5))

    valid = selective.dropna(subset=["risk"])

    ax.plot(
        valid["coverage"] * 100,
        valid["risk"] * 100,
        linewidth=2,
        label="Confidence-based selection",
    )

    ax.axhline(
        (1 - metrics["accuracy"]) * 100,
        linestyle="--",
        color="gray",
        label="Full-coverage error rate",
    )

    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Risk–Coverage Tradeoff")

    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)

    ax.grid(alpha=0.25)
    ax.legend()

    fig_selective.tight_layout()

    st.pyplot(fig_selective)
    plt.close(fig_selective)

    st.caption(
        "Higher confidence thresholds reduce coverage. "
        "Risk represents the error rate among accepted predictions. "
        "Results are descriptive estimates on the SST-2 validation set."
    )

    # --------------------------------------------------
    # 4. Confidence Threshold Comparison
    # --------------------------------------------------

    st.subheader("Confidence Threshold Comparison")

    thresholds = [0.5, 0.9, 0.95, 0.99, 0.995]

    comparison = []

    for threshold in thresholds:
        row = selective.iloc[
            (selective["threshold"] - threshold).abs().argmin()
        ]

        comparison.append({
            "Confidence Threshold": f"{threshold:.1%}",
            "Coverage": f"{row['coverage']:.2%}",
            "Error Rate": f"{row['risk']:.2%}",
            "Accuracy": f"{1 - row['risk']:.2%}",
            "Accepted Predictions": int(row["accepted"]),
        })

    st.dataframe(
        pd.DataFrame(comparison),
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------
    # 5. Confusion Matrix
    # --------------------------------------------------

    st.subheader("Confusion Matrix")

    matrix = metrics["confusion_matrix"]

    st.dataframe(
        pd.DataFrame(
            matrix,
            index=[
                "Actual Negative",
                "Actual Positive",
            ],
            columns=[
                "Predicted Negative",
                "Predicted Positive",
            ],
        ),
        width="stretch",
    )

    # --------------------------------------------------
    # 6. High-Confidence Mistakes
    # --------------------------------------------------

    st.subheader("Most Confident Mistakes")

    mistakes = predictions[
        predictions["true_label"]
        != predictions["predicted_label"]
    ].sort_values(
        "confidence",
        ascending=False,
    )

    st.dataframe(
        mistakes[
            [
                "sentence",
                "true_label",
                "predicted_label",
                "confidence",
            ]
        ].head(20),
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------
    # 7. Interpretation
    # --------------------------------------------------

    st.subheader("Key Findings")

    st.markdown(
        f"""
- **Classification accuracy:** {metrics['accuracy']:.2%}
- **Expected Calibration Error:** {diagnostics['ece']:.2%}
- **Average confidence:** {diagnostics['mean_confidence']:.2%}
- **High-confidence errors:** {diagnostics['high_confidence_errors']}

The classifier demonstrates strong overall accuracy but
exhibits overconfidence on this benchmark.

Selective prediction reduces the observed error rate
by abstaining on lower-confidence examples, at the
cost of reduced coverage.

These findings describe the evaluated dataset and
should not be interpreted as guarantees of performance
on unseen distributions.
"""
    )

    st.caption(
        "Source: rubrix/sst2_with_predictions. "
        "The repository documents a DistilBERT SST-2 pipeline. "
        "Saved predictions were matched against all 872 official "
        "SST-2 validation examples. Exact checkpoint provenance "
        "has not been independently verified."
    )
