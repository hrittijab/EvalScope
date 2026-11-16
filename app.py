
"""EvalScope interactive benchmark analysis dashboard."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from evalscope.data_loader import (
    load_experiment,
    summarize_experiment,
)


st.set_page_config(
    page_title="EvalScope | LLM Evaluation",
    page_icon="📊",
    layout="wide",
)

st.title("📊 EvalScope")
st.caption("Statistical Reliability of LLM Benchmarks")

st.sidebar.header("Dataset")

uploaded = st.sidebar.file_uploader(
    "Upload AgentSecBench experiment JSON",
    type=["json"],
)

default_path = Path("data/agentsecbench_experiment.json")


def load_uploaded_file(uploaded_file):
    """Validate uploaded JSON using the existing importer."""
    content = uploaded_file.getvalue()

    # Limit uploads to 20 MB.
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("Uploaded file exceeds 20 MB.")

    # Check JSON syntax before passing to the importer.
    json.loads(content.decode("utf-8"))

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "experiment.json"
        path.write_bytes(content)
        return load_experiment(path)


try:
    if uploaded is not None:
        df = load_uploaded_file(uploaded)
        dataset_name = uploaded.name
    elif default_path.exists():
        df = load_experiment(default_path)
        dataset_name = default_path.name
    else:
        st.info(
            "Upload an AgentSecBench experiment JSON "
            "to begin analyzing results."
        )
        st.stop()

except (ValueError, OSError, UnicodeError) as exc:
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

st.sidebar.success("Dataset loaded")
st.sidebar.caption(dataset_name)

summary = summarize_experiment(df)

st.subheader("Experiment Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Trials", summary["total_trials"])
col2.metric("Unique Attacks", summary["unique_attacks"])
col3.metric("Source Seeds", summary["source_attacks"])
col4.metric("Completed Trials", summary["completed_trials"])

st.divider()

st.subheader("Trial Completion")

completion = (
    df.groupby("mode")
    .agg(
        total=("attack_id", "size"),
        completed=("completed", "sum"),
    )
    .reset_index()
)

completion["incomplete"] = (
    completion["total"] - completion["completed"]
)

st.bar_chart(
    completion.set_index("mode")[["completed", "incomplete"]]
)

st.subheader("Explore Benchmark Records")

mode_options = sorted(df["mode"].unique().tolist())
category_options = sorted(df["category"].unique().tolist())

selected_modes = st.multiselect(
    "Defense configurations",
    options=mode_options,
    default=mode_options,
)

selected_categories = st.multiselect(
    "Attack categories",
    options=category_options,
    default=category_options,
)

filtered = df[
    df["mode"].isin(selected_modes)
    & df["category"].isin(selected_categories)
].copy()

st.caption(f"Showing {len(filtered)} of {len(df)} trials")

display_columns = [
    "attack_id",
    "source_attack_id",
    "category",
    "mode",
    "attack_success",
    "status",
    "completed",
    "duration_seconds",
]

st.dataframe(
    filtered[display_columns],
    use_container_width=True,
    hide_index=True,
)

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download filtered dataset (CSV)",
    data=csv_data,
    file_name="evalscope_filtered_trials.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "EvalScope | Experimental results are not guarantees "
    "of real-world model or defense performance."
)



# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

from evalscope.confidence import exact_binomial_interval
from evalscope.paired_analysis import compare_defenses
from evalscope.sensitivity import analyze_incomplete_trials


st.divider()
st.header("Statistical Analysis")

st.info(
    "Statistical results use the full uploaded dataset, "
    "independently of the record-explorer filters above."
)

analysis_modes = sorted(df["mode"].unique().tolist())

# ------------------------------------------------------------
# 1. ATTACK SUCCESS RATES
# ------------------------------------------------------------

st.subheader("1. Attack Success Rates")

rate_rows = []

for mode in analysis_modes:
    selected = df[
        (df["mode"] == mode)
        & (df["completed"])
    ]

    if selected.empty:
        continue

    successes = int(selected["attack_success"].sum())
    trials = len(selected)

    interval = exact_binomial_interval(
        successes,
        trials,
    )

    rate_rows.append({
        "Configuration": mode,
        "Completed Trials": trials,
        "Successful Attacks": successes,
        "Attack Success Rate (%)": round(
            interval["rate"] * 100, 2
        ),
        "95% CI Lower (%)": round(
            interval["ci_lower"] * 100, 2
        ),
        "95% CI Upper (%)": round(
            interval["ci_upper"] * 100, 2
        ),
    })

if rate_rows:
    rate_df = pd.DataFrame(rate_rows)

    st.dataframe(
        rate_df,
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        rate_df.set_index("Configuration")[
            "Attack Success Rate (%)"
        ]
    )

    st.caption(
        "Exact binomial confidence intervals assume "
        "independent trials. Shared attack seeds can "
        "violate this assumption."
    )

# ------------------------------------------------------------
# 2. PAIRED DEFENSE COMPARISON
# ------------------------------------------------------------

st.divider()
st.subheader("2. Paired Defense Comparison")

if len(analysis_modes) >= 2:

    col_a, col_b = st.columns(2)

    with col_a:
        mode_a = st.selectbox(
            "Reference configuration",
            analysis_modes,
            index=0,
            key="analysis_mode_a",
        )

    with col_b:
        remaining = [
            mode for mode in analysis_modes
            if mode != mode_a
        ]

        mode_b = st.selectbox(
            "Comparison configuration",
            remaining,
            key="analysis_mode_b",
        )

    n_resamples = st.slider(
        "Bootstrap resamples",
        min_value=500,
        max_value=10000,
        value=2000,
        step=500,
    )

    try:
        ordinary = compare_defenses(
            df,
            mode_a=mode_a,
            mode_b=mode_b,
            n_bootstrap=n_resamples,
            cluster=False,
        )

        clustered = compare_defenses(
            df,
            mode_a=mode_a,
            mode_b=mode_b,
            n_bootstrap=n_resamples,
            cluster=True,
        )

        st.metric(
            "Paired Attack Success Reduction",
            f"{ordinary['improvement'] * 100:.1f} pp",
        )

        st.metric(
            "Completed Attack Pairs",
            ordinary["n_pairs"],
        )

        comparison_df = pd.DataFrame([
            {
                "Method": "Ordinary paired bootstrap",
                "Lower CI (pp)": round(
                    ordinary["ci_lower"] * 100, 2
                ),
                "Upper CI (pp)": round(
                    ordinary["ci_upper"] * 100, 2
                ),
            },
            {
                "Method": "Seed-cluster bootstrap",
                "Lower CI (pp)": round(
                    clustered["ci_lower"] * 100, 2
                ),
                "Upper CI (pp)": round(
                    clustered["ci_upper"] * 100, 2
                ),
            },
        ])

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Positive improvement means the comparison "
            "configuration has a lower attack success rate. "
            "Cluster bootstrap resamples source attack seeds."
        )

    except ValueError as exc:
        st.warning(f"Comparison unavailable: {exc}")

else:
    st.warning(
        "At least two configurations are required "
        "for paired comparison."
    )

# ------------------------------------------------------------
# 3. INCOMPLETE TRIAL SENSITIVITY
# ------------------------------------------------------------

st.divider()
st.subheader("3. Incomplete-Trial Sensitivity")

if len(analysis_modes) >= 2:

    try:
        sensitivity = analyze_incomplete_trials(
            df,
            mode_a=mode_a,
            mode_b=mode_b,
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Attack Pairs",
            sensitivity["total_pairs"],
        )

        c2.metric(
            "Completed Pairs",
            sensitivity["completed_pairs"],
        )

        c3.metric(
            "Incomplete Pairs",
            sensitivity["incomplete_pairs"],
        )

        st.write("**Possible improvement under extreme assumptions**")

        lower = sensitivity[
            "worst_case_improvement"
        ] * 100

        upper = sensitivity[
            "best_case_improvement"
        ] * 100

        bounds_df = pd.DataFrame([
            {
                "Scenario": "Worst case",
                "Improvement (pp)": round(lower, 2),
            },
            {
                "Scenario": "Best case",
                "Improvement (pp)": round(upper, 2),
            },
        ])

        st.dataframe(
            bounds_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "These are logical sensitivity bounds over all "
            "attack pairs, not confidence intervals. "
            "Completed-only estimates use a different denominator."
        )

    except ValueError as exc:
        st.warning(f"Sensitivity analysis unavailable: {exc}")

st.divider()
st.caption("EvalScope — Reproducible LLM Benchmark Evaluation")



# ============================================================
# RANKING STABILITY DASHBOARD
# ============================================================

from evalscope.ranking import analyze_ranking_stability

st.divider()
st.header("Ranking Stability")

st.write(
    "How consistently does each configuration rank best "
    "when benchmark attacks are resampled?"
)

available_modes = sorted(df["mode"].unique().tolist())

if len(available_modes) < 2:
    st.warning(
        "Ranking analysis requires at least two configurations."
    )

else:
    selected_ranking_modes = st.multiselect(
        "Select configurations to compare",
        options=available_modes,
        default=available_modes,
        key="ranking_modes",
    )

    ranking_resamples = st.slider(
        "Ranking bootstrap resamples",
        min_value=500,
        max_value=10000,
        value=2000,
        step=500,
        key="ranking_resamples",
    )

    ranking_method = st.radio(
        "Resampling method",
        options=[
            "Seed-cluster bootstrap",
            "Ordinary bootstrap",
        ],
        index=0,
        horizontal=True,
    )

    if len(selected_ranking_modes) < 2:
        st.info("Select at least two configurations.")

    else:
        try:
            ranking = analyze_ranking_stability(
                df,
                modes=selected_ranking_modes,
                n_bootstrap=ranking_resamples,
                seed=42,
                cluster=(
                    ranking_method == "Seed-cluster bootstrap"
                ),
            )

            st.subheader("Benchmark Coverage")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Common Completed Attacks",
                ranking["n_common_attacks"],
            )

            col2.metric(
                "Excluded Attacks",
                ranking["n_excluded_attacks"],
            )

            col3.metric(
                "Source Clusters",
                ranking["n_source_clusters"],
            )

            st.subheader("Probability of Ranking Best")

            ranking_rows = []

            for mode in ranking["modes"]:
                ranking_rows.append({
                    "Configuration": mode,
                    "Best-Rank Frequency (%)": round(
                        ranking["best_rank_probability"][mode]
                        * 100,
                        2,
                    ),
                    "Average Rank": round(
                        ranking["average_rank"][mode],
                        2,
                    ),
                    "Observed ASR (%)": round(
                        ranking["observed_success_rates"][mode]
                        * 100,
                        2,
                    ),
                })

            ranking_df = pd.DataFrame(ranking_rows)

            st.bar_chart(
                ranking_df.set_index("Configuration")[
                    "Best-Rank Frequency (%)"
                ]
            )

            st.dataframe(
                ranking_df,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Pairwise Ranking Comparisons")

            pairwise_rows = []

            for comparison, probability in ranking[
                "pairwise_win_probability"
            ].items():
                pairwise_rows.append({
                    "Comparison": comparison,
                    "Win Credit (%)": round(
                        probability * 100,
                        2,
                    ),
                })

            st.dataframe(
                pd.DataFrame(pairwise_rows),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Lower attack success rate ranks better. "
                "Ties split ranking credit equally. "
                "These frequencies describe bootstrap "
                "resamples of the observed benchmark, "
                "not calibrated probabilities of "
                "future superiority."
            )

            # ----------------------------------------
            # Download ranking results
            # ----------------------------------------

            import json

            ranking_json = json.dumps(
                ranking,
                indent=2,
            )

            st.download_button(
                "Download Ranking Analysis (JSON)",
                data=ranking_json,
                file_name="evalscope_ranking_analysis.json",
                mime="application/json",
            )

        except ValueError as exc:
            st.warning(f"Ranking unavailable: {exc}")
