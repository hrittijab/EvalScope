
"""Generate reproducible benchmark reliability visualizations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evalscope.confidence import exact_binomial_interval
from evalscope.paired_analysis import compare_defenses
from evalscope.sensitivity import analyze_incomplete_trials


def plot_evaluation_summary(
    df: pd.DataFrame,
    output_dir: str | Path = "results",
    mode_a: str = "baseline",
    mode_b: str = "defended",
) -> list[Path]:
    """Save three statistical evaluation figures."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []

    # Figure 1: observed success rates and binomial intervals.
    modes = [mode_a, mode_b]
    intervals = []

    for mode in modes:
        selected = df[
            (df["mode"] == mode) & (df["completed"])
        ]

        if selected.empty:
            raise ValueError(f"No completed trials for {mode}")

        intervals.append(
            exact_binomial_interval(
                int(selected["attack_success"].sum()),
                len(selected),
            )
        )

    rates = np.array([x["rate"] for x in intervals])
    lower = np.array([x["ci_lower"] for x in intervals])
    upper = np.array([x["ci_upper"] for x in intervals])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        range(2),
        rates * 100,
        yerr=[
            (rates - lower) * 100,
            (upper - rates) * 100,
        ],
        fmt="o",
        markersize=9,
        capsize=7,
        linewidth=2,
    )

    ax.set_xticks(range(2), modes)
    ax.set_ylabel("Attack success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Observed Attack Success and 95% Binomial CIs")
    ax.grid(axis="y", alpha=0.2)

    for i, item in enumerate(intervals):
        ax.annotate(
            f"{item['successes']}/{item['trials']}",
            (i, rates[i] * 100),
            xytext=(12, 10),
            textcoords="offset points",
        )

    fig.tight_layout()

    path = output_dir / "attack_success_intervals.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    # Figure 2: paired bootstrap comparison.
    ordinary = compare_defenses(
        df, mode_a, mode_b, cluster=False
    )
    clustered = compare_defenses(
        df, mode_a, mode_b, cluster=True
    )

    comparisons = [ordinary, clustered]
    labels = ["Paired bootstrap", "Seed-cluster bootstrap"]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, result in enumerate(comparisons):
        estimate = result["improvement"] * 100
        lo = result["ci_lower"] * 100
        hi = result["ci_upper"] * 100

        ax.errorbar(
            estimate,
            i,
            xerr=[[estimate - lo], [hi - estimate]],
            fmt="o",
            markersize=8,
            capsize=6,
        )

    ax.axvline(0, linestyle="--", alpha=0.5)
    ax.set_yticks(range(2), labels)
    ax.invert_yaxis()
    ax.set_xlim(-10, 100)
    ax.set_xlabel("Reduction in attack success (percentage points)")
    ax.set_title(
        f"Paired Improvement — {ordinary['n_pairs']} Completed Pairs"
    )
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()

    path = output_dir / "paired_improvement.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    # Figure 3: incomplete-trial sensitivity.
    sensitivity = analyze_incomplete_trials(
        df, mode_a, mode_b
    )

    low = sensitivity["worst_case_improvement"] * 100
    high = sensitivity["best_case_improvement"] * 100
    completed = sensitivity["completed_only_improvement"]

    fig, ax = plt.subplots(figsize=(9, 3.5))

    ax.plot(
        [low, high],
        [0, 0],
        linewidth=6,
        solid_capstyle="round",
    )

    ax.scatter([low, high], [0, 0], s=80, zorder=3)

    if completed is not None:
        ax.scatter(
            [completed * 100],
            [0],
            s=130,
            marker="D",
            zorder=4,
            label="Completed-only estimate",
        )
        ax.legend(loc="upper center")

    ax.set_xlim(-10, 100)
    ax.set_ylim(-0.5, 0.7)
    ax.set_yticks([])
    ax.set_xlabel("Reduction in attack success (percentage points)")
    ax.set_title(
        "Incomplete-Trial Sensitivity Bounds "
        f"({sensitivity['total_pairs']} Pairs)"
    )
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()

    path = output_dir / "incomplete_trial_sensitivity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths
