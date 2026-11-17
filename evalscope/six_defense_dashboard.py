
"""Streamlit dashboard for the six-defense case study."""

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from evalscope.six_defense_loader import (
    load_six_defense_attacks,
    load_six_defense_benign,
    security_utility_summary,
)
from evalscope.six_defense_statistics import (
    paired_comparison,
    add_holm_correction,
)


def render_six_defense_dashboard():
    st.header("Six-Defense Security–Utility Evaluation")

    data_dir = Path(__file__).resolve().parents[1] / "data"

    attack_path = data_dir / "six_defense_comparison.json"
    benign_path = data_dir / "benign_six_defense_comparison.json"

    if not attack_path.exists() or not benign_path.exists():
        st.info(
            "Six-defense benchmark files are missing from data/. "
            "Copy the AgentSecBench JSON results into this folder."
        )
        return

    attacks = load_six_defense_attacks(attack_path)
    benign = load_six_defense_benign(benign_path)

    summary = security_utility_summary(attacks, benign)

    st.caption(
        f"{len(attacks)} attack trials and {len(benign)} "
        "benign-task trials across six defenses."
    )

    st.subheader("Security and Utility")

    display = summary[
        [
            "defense",
            "attack_success_rate",
            "benign_success_rate",
        ]
    ].copy()

    display["attack_success_rate"] *= 100
    display["benign_success_rate"] *= 100

    st.dataframe(
        display.rename(
            columns={
                "defense": "Defense",
                "attack_success_rate": "Attack success (%)",
                "benign_success_rate": "Benign success (%)",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in display.iterrows():
        ax.scatter(
            row["attack_success_rate"],
            row["benign_success_rate"],
            s=100,
        )
        ax.annotate(
            row["defense"],
            (
                row["attack_success_rate"],
                row["benign_success_rate"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("Attack success (%) — lower is better")
    ax.set_ylabel("Benign success (%) — higher is better")
    ax.set_title("Security–Utility Tradeoff")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Paired Statistical Comparisons")

    dataset = st.radio(
        "Select evaluation type",
        ["Attack resistance", "Benign utility"],
        horizontal=True,
    )

    selected = (
        attacks if dataset == "Attack resistance" else benign
    )

    comparisons = add_holm_correction(
        paired_comparison(selected)
    )

    st.dataframe(
        comparisons.round(4),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Differences are baseline success minus defense success. "
        "Positive differences indicate improved attack resistance "
        "for attacks, but reduced task utility for benign cases. "
        "Bootstrap intervals are unadjusted. Holm correction "
        "controls family-wise error across five comparisons."
    )

    st.warning(
        "This is an exploratory, single-model benchmark. "
        "No attack comparison meets the 0.05 significance "
        "threshold after Holm correction. Non-significant "
        "benign comparisons do not establish equivalent utility."
    )
