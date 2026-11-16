
"""Benchmark ranking stability using paired cluster bootstrap."""

from itertools import combinations

import numpy as np
import pandas as pd


def analyze_ranking_stability(
    df: pd.DataFrame,
    modes: list[str] | None = None,
    n_bootstrap: int = 5000,
    seed: int = 42,
    cluster: bool = True,
) -> dict:
    """
    Estimate how frequently each configuration ranks best.

    All configurations are evaluated on the same completed
    attack IDs within one experiment.

    Tied winners split credit equally.
    """
    if modes is None:
        modes = sorted(df["mode"].unique().tolist())

    if len(modes) < 2 or len(set(modes)) != len(modes):
        raise ValueError("At least two distinct modes are required.")

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")

    selected = df[df["mode"].isin(modes)].copy()

    if selected.empty:
        raise ValueError("No matching evaluation records.")

    experiments = selected["experiment_id"].dropna().unique()

    if len(experiments) != 1:
        raise ValueError(
            "Ranking requires exactly one experiment."
        )

    if selected.duplicated(["attack_id", "mode"]).any():
        raise ValueError("Duplicate attack/mode records.")

    completed = selected[selected["completed"]].copy()

    wide = completed.pivot(
        index="attack_id",
        columns="mode",
        values="attack_success",
    )

    wide = wide.reindex(columns=modes).dropna()

    if wide.empty:
        raise ValueError(
            "No attacks completed under every selected mode."
        )

    metadata = selected.drop_duplicates("attack_id").set_index(
        "attack_id"
    )

    seeds = metadata.loc[
        wide.index, "source_attack_id"
    ]

    values = wide.to_numpy(dtype=float)
    observed_rates = values.mean(axis=0)

    rng = np.random.default_rng(seed)
    winner_credit = np.zeros(len(modes))
    rank_sums = np.zeros(len(modes))

    pairwise_wins = {
        f"{a}_over_{b}": 0.0
        for a, b in combinations(modes, 2)
    }

    if cluster:
        groups = [
            np.flatnonzero(seeds.to_numpy() == source)
            for source in sorted(seeds.unique())
        ]
    else:
        groups = [
            np.array([i]) for i in range(len(wide))
        ]

    for _ in range(n_bootstrap):
        sampled_groups = rng.integers(
            0, len(groups), size=len(groups)
        )

        indices = np.concatenate(
            [groups[i] for i in sampled_groups]
        )

        rates = values[indices].mean(axis=0)

        # Lower attack success is better.
        best = np.flatnonzero(
            np.isclose(rates, rates.min(), atol=1e-12)
        )

        winner_credit[best] += 1 / len(best)

        # Average ranks handle ties.
        ranks = pd.Series(rates).rank(
            method="average", ascending=True
        ).to_numpy()

        rank_sums += ranks

        for i, j in combinations(range(len(modes)), 2):
            key = f"{modes[i]}_over_{modes[j]}"

            if np.isclose(rates[i], rates[j], atol=1e-12):
                pairwise_wins[key] += 0.5
            elif rates[i] < rates[j]:
                pairwise_wins[key] += 1.0

    return {
        "modes": modes,
        "n_common_attacks": len(wide),
        "n_excluded_attacks": (
            selected["attack_id"].nunique() - len(wide)
        ),
        "n_source_clusters": int(seeds.nunique()),
        "method": (
            "cluster_bootstrap" if cluster else "ordinary_bootstrap"
        ),
        "observed_success_rates": {
            mode: float(observed_rates[i])
            for i, mode in enumerate(modes)
        },
        "best_rank_probability": {
            mode: float(winner_credit[i] / n_bootstrap)
            for i, mode in enumerate(modes)
        },
        "average_rank": {
            mode: float(rank_sums[i] / n_bootstrap)
            for i, mode in enumerate(modes)
        },
        "pairwise_win_probability": {
            key: float(value / n_bootstrap)
            for key, value in pairwise_wins.items()
        },
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }
