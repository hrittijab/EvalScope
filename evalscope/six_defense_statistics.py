
"""Paired statistical comparisons for six-defense benchmarks."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from evalscope.six_defense_loader import DEFENSES


def paired_comparison(
    df,
    reference="baseline",
    n_bootstrap=5000,
    seed=42,
):
    """Compare each defense with a reference on matched cases.

    Difference = reference success rate - defense success rate.
    For attacks, positive values indicate improvement.
    For benign tasks, positive values indicate utility loss.
    """
    completed = df[df["completed"]].copy()

    if completed.duplicated(["case_id", "defense"]).any():
        raise ValueError(
            "Multiple trials per case/defense require explicit handling."
        )

    wide = completed.pivot(
        index="case_id",
        columns="defense",
        values="success",
    )

    if reference not in wide.columns:
        raise ValueError("Reference defense is missing.")

    rng = np.random.default_rng(seed)
    results = []

    for defense in DEFENSES:
        if defense == reference:
            continue

        if defense not in wide.columns:
            raise ValueError(f"Missing defense: {defense}")

        matched = wide[[reference, defense]].dropna()

        if matched.empty:
            raise ValueError(f"No matched cases for {defense}")

        ref = matched[reference].to_numpy(dtype=int)
        alt = matched[defense].to_numpy(dtype=int)

        differences = ref - alt
        observed = float(differences.mean())

        indices = rng.integers(
            0,
            len(differences),
            size=(n_bootstrap, len(differences)),
        )

        samples = differences[indices].mean(axis=1)
        lower, upper = np.quantile(samples, [0.025, 0.975])

        ref_only = int(((ref == 1) & (alt == 0)).sum())
        alt_only = int(((ref == 0) & (alt == 1)).sum())
        discordant = ref_only + alt_only

        p_value = (
            min(
                1.0,
                2 * binomtest(
                    min(ref_only, alt_only),
                    discordant,
                    0.5,
                    alternative="less",
                ).pvalue,
            )
            if discordant
            else 1.0
        )

        results.append({
            "reference": reference,
            "defense": defense,
            "matched_cases": len(matched),
            "difference": observed,
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "ref_only_success": ref_only,
            "defense_only_success": alt_only,
            "mcnemar_p": p_value,
        })

    return pd.DataFrame(results)


def holm_adjust(p_values):
    """Holm step-down family-wise error correction."""
    p = np.asarray(p_values, dtype=float)

    if p.ndim != 1 or np.any(~np.isfinite(p)):
        raise ValueError("Expected finite one-dimensional p-values.")

    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be between 0 and 1.")

    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)

    running_max = 0.0
    m = len(p)

    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max

    return adjusted


def add_holm_correction(results):
    """Add Holm-adjusted p-values to a comparison table."""
    results = results.copy()
    results["holm_p"] = holm_adjust(results["mcnemar_p"])
    results["significant_05"] = results["holm_p"] < 0.05
    return results
