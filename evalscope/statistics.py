
"""Statistical estimation for benchmark outcomes."""

import numpy as np
import pandas as pd


def bootstrap_success_rate(
    df: pd.DataFrame,
    mode: str,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
    cluster: bool = False,
) -> dict:
    """
    Estimate attack success rate and a percentile bootstrap CI.

    Only completed, valid trials are included.
    When cluster=True, resample source_attack_id clusters.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    selected = df[
        (df["mode"] == mode) &
        (df["completed"])
    ].copy()

    if selected.empty:
        raise ValueError(f"No completed trials for mode: {mode}")

    outcomes = selected["attack_success"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap)

    if cluster:
        if selected["source_attack_id"].isna().any():
            raise ValueError("Missing source attack IDs.")

        groups = [
            group["attack_success"].to_numpy(dtype=float)
            for _, group in selected.groupby("source_attack_id")
        ]

        for i in range(n_bootstrap):
            indices = rng.integers(0, len(groups), len(groups))
            sample = np.concatenate([groups[j] for j in indices])
            estimates[i] = sample.mean()

    else:
        n = len(outcomes)

        for i in range(n_bootstrap):
            sample = rng.choice(outcomes, size=n, replace=True)
            estimates[i] = sample.mean()

    alpha = (1 - confidence) / 2

    lower, upper = np.quantile(
        estimates,
        [alpha, 1 - alpha],
    )

    return {
        "mode": mode,
        "method": "cluster_bootstrap" if cluster else "bootstrap",
        "n_completed": len(selected),
        "n_clusters": (
            selected["source_attack_id"].nunique()
            if cluster else None
        ),
        "success_rate": float(outcomes.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }
