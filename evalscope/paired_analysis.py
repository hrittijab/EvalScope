
"""Paired statistical comparisons for benchmark configurations."""

import math

import numpy as np
import pandas as pd

from evalscope.data_loader import paired_trials


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    """Two-sided exact McNemar test using the discordant pairs."""
    if b < 0 or c < 0:
        raise ValueError("Discordant counts must be nonnegative.")

    n = b + c

    if n == 0:
        return 1.0

    k = min(b, c)

    probability = sum(
        math.comb(n, i) for i in range(k + 1)
    ) / (2 ** n)

    return min(1.0, 2 * probability)


def compare_defenses(
    df: pd.DataFrame,
    mode_a: str = "baseline",
    mode_b: str = "defended",
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
    cluster: bool = False,
) -> dict:
    """Compare paired attack success rates between two configurations."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    pairs = paired_trials(
        df,
        mode_a=mode_a,
        mode_b=mode_b,
        completed_only=True,
    )

    if pairs.empty:
        raise ValueError("No fully completed attack pairs found.")

    a = pairs["attack_success_a"].to_numpy(dtype=int)
    b = pairs["attack_success_b"].to_numpy(dtype=int)

    differences = a - b

    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap)

    if cluster:
        groups = [
            group["attack_success_a"].to_numpy(dtype=int)
            - group["attack_success_b"].to_numpy(dtype=int)
            for _, group in pairs.groupby("source_attack_id_a")
        ]

        for i in range(n_bootstrap):
            indices = rng.integers(0, len(groups), len(groups))
            sample = np.concatenate([groups[j] for j in indices])
            estimates[i] = sample.mean()

    else:
        for i in range(n_bootstrap):
            sample = rng.choice(
                differences,
                size=len(differences),
                replace=True,
            )
            estimates[i] = sample.mean()

    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(
        estimates,
        [alpha, 1 - alpha],
    )

    return {
        "mode_a": mode_a,
        "mode_b": mode_b,
        "n_pairs": len(pairs),
        "n_clusters": int(pairs["source_attack_id_a"].nunique()),
        "rate_a": float(a.mean()),
        "rate_b": float(b.mean()),
        "improvement": float(differences.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "mcnemar_pvalue": exact_mcnemar_pvalue(a_only, b_only),
        "a_only_success": a_only,
        "b_only_success": b_only,
        "method": "cluster_bootstrap" if cluster else "paired_bootstrap",
        "confidence": confidence,
    }
