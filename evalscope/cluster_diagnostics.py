
"""Cluster-level robustness diagnostics for paired experiments."""

import numpy as np
import pandas as pd

from evalscope.data_loader import paired_trials


def leave_one_cluster_out(
    df: pd.DataFrame,
    mode_a: str = "baseline",
    mode_b: str = "defended",
) -> dict:
    """
    Evaluate sensitivity of paired improvement to excluding
    one source-attack cluster at a time.

    Positive improvement means mode_b has a lower
    attack success rate than mode_a.
    """

    pairs = paired_trials(
        df,
        mode_a=mode_a,
        mode_b=mode_b,
        completed_only=True,
    ).copy()

    if pairs.empty:
        raise ValueError(
            "No completed paired trials available."
        )

    required = {
        "source_attack_id_a",
        "source_attack_id_b",
        "attack_success_a",
        "attack_success_b",
    }

    missing = required - set(pairs.columns)

    if missing:
        raise ValueError(
            f"Missing paired columns: {sorted(missing)}"
        )

    if (
        pairs["source_attack_id_a"].isna().any()
        or pairs["source_attack_id_b"].isna().any()
    ):
        raise ValueError(
            "Missing source attack identifiers."
        )

    if not (
        pairs["source_attack_id_a"]
        == pairs["source_attack_id_b"]
    ).all():
        raise ValueError(
            "Mismatched source attack identifiers "
            "between paired configurations."
        )

    pairs["improvement"] = (
        pairs["attack_success_a"].astype(float)
        - pairs["attack_success_b"].astype(float)
    )

    observed = float(
        pairs["improvement"].mean()
    )

    clusters = sorted(
        pairs["source_attack_id_a"].unique(),
        key=str,
    )

    if len(clusters) < 2:
        raise ValueError(
            "At least two source clusters are required."
        )

    estimates = []

    for cluster in clusters:
        remaining = pairs[
            pairs["source_attack_id_a"] != cluster
        ]

        estimate = float(
            remaining["improvement"].mean()
        )

        estimates.append({
            "excluded_cluster": str(cluster),
            "remaining_pairs": int(len(remaining)),
            "improvement": estimate,
            "change_from_observed": estimate - observed,
        })

    values = np.array(
        [item["improvement"] for item in estimates],
        dtype=float,
    )

    return {
        "mode_a": mode_a,
        "mode_b": mode_b,
        "n_pairs": int(len(pairs)),
        "n_clusters": len(clusters),
        "observed_improvement": observed,
        "min_loco_improvement": float(values.min()),
        "max_loco_improvement": float(values.max()),
        "max_absolute_change": float(
            np.max(np.abs(values - observed))
        ),
        "all_loco_positive": bool(
            np.all(values > 0)
        ),
        "cluster_results": estimates,
    }
