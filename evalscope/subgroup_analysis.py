
"""Subgroup effectiveness analysis for paired security benchmarks."""

import numpy as np
import pandas as pd

from evalscope.data_loader import paired_trials


def analyze_subgroups(
    df: pd.DataFrame,
    mode_a: str = "baseline",
    mode_b: str = "defended",
    group_column: str = "category",
    n_bootstrap: int = 2000,
    seed: int = 42,
    min_pairs: int = 10,
    min_clusters: int = 5,
) -> pd.DataFrame:
    """
    Compare attack success rates across subgroups.

    Improvement is ASR(mode_a) - ASR(mode_b).
    Positive values indicate lower attack success in mode_b.

    Bootstrap resampling uses source-attack clusters.
    """

    if mode_a == mode_b:
        raise ValueError("Select two different configurations.")

    if group_column not in df.columns:
        raise ValueError(
            f"Grouping column not found: {group_column}"
        )

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")

    if min_pairs < 1 or min_clusters < 1:
        raise ValueError("Minimum thresholds must be positive.")

    pairs = paired_trials(
        df,
        mode_a=mode_a,
        mode_b=mode_b,
        completed_only=True,
    ).copy()

    if pairs.empty:
        raise ValueError("No completed paired trials available.")

    group_a = f"{group_column}_a"
    group_b = f"{group_column}_b"

    if group_a not in pairs.columns or group_b not in pairs.columns:
        raise ValueError(
            "Paired data does not contain the grouping column."
        )

    if pairs[group_a].isna().any() or pairs[group_b].isna().any():
        raise ValueError("Subgroup labels cannot be missing.")

    if not (pairs[group_a] == pairs[group_b]).all():
        raise ValueError(
            "Subgroup labels disagree between matched trials."
        )

    source_a = "source_attack_id_a"
    source_b = "source_attack_id_b"

    if not (pairs[source_a] == pairs[source_b]).all():
        raise ValueError(
            "Source cluster IDs disagree between matched trials."
        )

    rng = np.random.default_rng(seed)
    rows = []

    for group_value, group in pairs.groupby(group_a, sort=True):
        group = group.copy()

        a = group["attack_success_a"].astype(float)
        b = group["attack_success_b"].astype(float)

        baseline_asr = float(a.mean())
        defended_asr = float(b.mean())
        improvement = baseline_asr - defended_asr

        clusters = [
            cluster_df
            for _, cluster_df in group.groupby(
                source_a,
                sort=True,
            )
        ]

        n_clusters = len(clusters)
        n_pairs = len(group)

        ci_lower = np.nan
        ci_upper = np.nan

        if n_clusters >= 2:
            estimates = np.empty(n_bootstrap, dtype=float)

            for i in range(n_bootstrap):
                sampled = rng.integers(
                    0,
                    n_clusters,
                    size=n_clusters,
                )

                sampled_groups = [
                    clusters[j]
                    for j in sampled
                ]

                bootstrap_df = pd.concat(
                    sampled_groups,
                    ignore_index=True,
                )

                estimates[i] = (
                    bootstrap_df["attack_success_a"].mean()
                    - bootstrap_df["attack_success_b"].mean()
                )

            ci_lower, ci_upper = np.quantile(
                estimates,
                [0.025, 0.975],
            )

        warnings = []

        if n_pairs < min_pairs:
            warnings.append("Few completed pairs")

        if n_clusters < min_clusters:
            warnings.append("Few source clusters")

        if n_clusters < 2:
            warnings.append(
                "Cluster bootstrap unavailable"
            )

        rows.append(
            {
                "category": str(group_value),
                "n_pairs": int(n_pairs),
                "n_clusters": int(n_clusters),
                "baseline_asr": baseline_asr,
                "defended_asr": defended_asr,
                "improvement": float(improvement),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "warning": "; ".join(warnings),
            }
        )

    if not rows:
        raise ValueError("No subgroups available for analysis.")

    return pd.DataFrame(rows).sort_values(
        "category"
    ).reset_index(drop=True)
