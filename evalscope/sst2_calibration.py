
"""Confidence calibration diagnostics for SST-2."""

import numpy as np
import matplotlib.pyplot as plt


def calibration_analysis(predictions, n_bins=10):
    confidence = predictions["confidence"].to_numpy()
    correct = (
        predictions["true_label"].to_numpy()
        == predictions["predicted_label"].to_numpy()
    ).astype(int)

    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.minimum(
        np.digitize(confidence, edges[1:-1]),
        n_bins - 1,
    )

    mean_confidence = []
    observed_accuracy = []
    bin_counts = []
    ece = 0.0

    for i in range(n_bins):
        mask = bins == i
        count = int(mask.sum())

        if count == 0:
            continue

        avg_conf = float(confidence[mask].mean())
        avg_acc = float(correct[mask].mean())

        mean_confidence.append(avg_conf)
        observed_accuracy.append(avg_acc)
        bin_counts.append(count)

        ece += (count / len(confidence)) * abs(
            avg_acc - avg_conf
        )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        [0, 1], [0, 1],
        "--", color="gray",
        label="Perfect calibration",
    )

    ax.plot(
        mean_confidence,
        observed_accuracy,
        "o-",
        label="Observed calibration",
    )

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("SST-2 Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    high_confidence = confidence >= 0.99

    diagnostics = {
        "ece": float(ece),
        "mean_confidence": float(confidence.mean()),
        "observed_accuracy": float(correct.mean()),
        "high_confidence_predictions": int(
            high_confidence.sum()
        ),
        "high_confidence_errors": int(
            ((correct == 0) & high_confidence).sum()
        ),
    }

    return diagnostics, fig
