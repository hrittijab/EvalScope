
"""Analyze published SST-2 sentiment predictions."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)


def analyze_sst2_predictions(df):
    """Calculate accuracy and probability-based metrics."""
    rows = []

    for _, record in df.iterrows():
        scores = {
            item["label"].upper(): float(item["score"])
            for item in record["prediction"]
        }

        if set(scores) != {"POSITIVE", "NEGATIVE"}:
            raise ValueError("Expected positive and negative scores.")

        true_label = record["annotation"].upper()

        if true_label not in {"POSITIVE", "NEGATIVE"}:
            raise ValueError("Unknown ground-truth label.")

        predicted_label = max(scores, key=scores.get)

        rows.append({
            "sentence": record["inputs"]["text"],
            "true_label": int(true_label == "POSITIVE"),
            "predicted_label": int(predicted_label == "POSITIVE"),
            "positive_probability": scores["POSITIVE"],
            "confidence": max(scores.values()),
        })

    result = pd.DataFrame(rows)

    y_true = result["true_label"].to_numpy()
    y_pred = result["predicted_label"].to_numpy()
    probabilities = result["positive_probability"].to_numpy()

    correct = int((y_true == y_pred).sum())
    total = len(result)

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

    return result, metrics
