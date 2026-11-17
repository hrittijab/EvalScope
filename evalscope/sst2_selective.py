import numpy as np
import pandas as pd


def selective_prediction_analysis(predictions):
    df = predictions.copy()

    df["error"] = (
        df["true_label"] != df["predicted_label"]
    ).astype(int)

    thresholds = np.linspace(0.5, 1.0, 101)

    rows = []

    for threshold in thresholds:
        accepted = df[df["confidence"] >= threshold]

        coverage = len(accepted) / len(df)

        risk = (
            accepted["error"].mean()
            if len(accepted) > 0
            else np.nan
        )

        rows.append({
            "threshold": threshold,
            "coverage": coverage,
            "risk": risk,
            "accepted": len(accepted),
        })

    return pd.DataFrame(rows)