
"""EvalScope interactive benchmark analysis dashboard."""

import json
import hashlib
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

# Fingerprint the actual dataset, not only its experiment ID.
# This prevents showing stale results after a different upload.
dataset_fingerprint = hashlib.sha256(
    pd.util.hash_pandas_object(df, index=True).values.tobytes()
).hexdigest()

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


# ============================================================
# CLUSTER ROBUSTNESS ANALYSIS
# ============================================================

from evalscope.cluster_diagnostics import leave_one_cluster_out

st.divider()
st.header("Cluster Robustness Analysis")

st.write(
    "Evaluate whether the observed difference between "
    "two configurations is sensitive to individual "
    "source-attack clusters."
)

st.caption(
    "Leave-One-Cluster-Out (LOCO) analysis removes one "
    "source attack seed at a time and recalculates the "
    "paired attack-success reduction."
)

if len(analysis_modes) >= 2:

    if st.button("Run Cluster Robustness Analysis"):

        try:
            with st.spinner("Analyzing source clusters..."):

                loco = leave_one_cluster_out(
                    df,
                    mode_a=mode_a,
                    mode_b=mode_b,
                )

            st.session_state["loco_results"] = loco
            st.session_state["loco_modes"] = (
                mode_a,
                mode_b,
                dataset_fingerprint,
            )

        except ValueError as exc:
            st.error(
                f"Cluster analysis unavailable: {exc}"
            )

    if (
        "loco_results" in st.session_state
        and st.session_state.get("loco_modes")
        == (mode_a, mode_b, dataset_fingerprint)
    ):

        loco = st.session_state["loco_results"]

        st.subheader("Robustness Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Completed Pairs",
                loco["n_pairs"],
            )

        with col2:
            st.metric(
                "Source Clusters",
                loco["n_clusters"],
            )

        with col3:
            st.metric(
                "Observed Improvement",
                f"{loco['observed_improvement'] * 100:.2f} pp",
            )

        with col4:
            st.metric(
                "Largest LOCO Change",
                f"{loco['max_absolute_change'] * 100:.2f} pp",
            )

        st.subheader("Improvement Stability")

        st.write(
            f"After excluding individual source clusters, "
            f"the estimated reduction ranged from "
            f"**{loco['min_loco_improvement'] * 100:.2f}** "
            f"to "
            f"**{loco['max_loco_improvement'] * 100:.2f} "
            f"percentage points**."
        )

        if loco["all_loco_positive"]:
            st.success(
                "The estimated improvement remained "
                "positive after excluding every individual "
                "source cluster."
            )
        else:
            st.warning(
                "At least one cluster exclusion produced "
                "a non-positive improvement estimate."
            )

        # --------------------------------------------
        # Cluster-level results
        # --------------------------------------------

        cluster_df = pd.DataFrame(
            loco["cluster_results"]
        )

        cluster_df["Improvement (pp)"] = (
            cluster_df["improvement"] * 100
        )

        cluster_df["Change (pp)"] = (
            cluster_df["change_from_observed"] * 100
        )

        st.subheader(
            "Leave-One-Cluster-Out Estimates"
        )

        st.caption(
            "Each point represents the estimated reduction "
            "after removing one source attack seed."
        )

        # --------------------------------------------
        # Visualization
        # --------------------------------------------

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        chart_df = cluster_df.sort_values(
            "Improvement (pp)"
        )

        ax.scatter(
            chart_df["excluded_cluster"],
            chart_df["Improvement (pp)"],
            color="#2563eb",
            s=45,
            label="LOCO estimate",
        )

        ax.axhline(
            y=loco["observed_improvement"] * 100,
            color="#dc2626",
            linestyle="--",
            linewidth=2,
            label="Observed improvement",
        )

        ax.set_xlabel("Excluded Source Cluster")
        ax.set_ylabel("Attack Success Reduction (pp)")

        ax.set_title(
            "Sensitivity to Individual Source Clusters"
        )

        ax.tick_params(
            axis="x",
            rotation=90,
        )

        ax.legend()
        ax.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # --------------------------------------------
        # Detailed table
        # --------------------------------------------

        with st.expander(
            "View Cluster-Level Results"
        ):

            st.dataframe(
                cluster_df[
                    [
                        "excluded_cluster",
                        "remaining_pairs",
                        "Improvement (pp)",
                        "Change (pp)",
                    ]
                ].rename(
                    columns={
                        "excluded_cluster":
                            "Excluded Cluster",
                        "remaining_pairs":
                            "Remaining Pairs",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        # --------------------------------------------
        # CSV download
        # --------------------------------------------

        csv_data = cluster_df.to_csv(
            index=False
        )

        st.download_button(
            "Download Cluster Robustness Results (CSV)",
            data=csv_data,
            file_name="evalscope_cluster_robustness.csv",
            mime="text/csv",
        )

        st.caption(
            "LOCO is a sensitivity diagnostic, not a "
            "confidence interval or formal significance "
            "test. Stability within the observed benchmark "
            "does not establish generalization to unseen "
            "attacks."
        )

else:
    st.info(
        "Upload a dataset with at least two configurations "
        "to run cluster robustness analysis."
    )



# ============================================================
# SUBGROUP EFFECTIVENESS ANALYSIS
# ============================================================

from evalscope.subgroup_analysis import analyze_subgroups
import matplotlib.pyplot as plt

st.divider()
st.header("Subgroup Effectiveness Analysis")

st.write(
    "Investigate how defense effectiveness varies across "
    "attack categories using matched evaluations and "
    "source-cluster bootstrap confidence intervals."
)

if len(analysis_modes) >= 2:

    subgroup_bootstrap = st.slider(
        "Subgroup bootstrap resamples",
        min_value=200,
        max_value=5000,
        value=2000,
        step=200,
        key="subgroup_bootstrap",
    )

    if st.button("Run Subgroup Analysis"):

        try:
            with st.spinner("Analyzing attack categories..."):
                subgroup_results = analyze_subgroups(
                    df,
                    mode_a=mode_a,
                    mode_b=mode_b,
                    group_column="category",
                    n_bootstrap=subgroup_bootstrap,
                    seed=42,
                )

            st.session_state["subgroup_results"] = (
                subgroup_results
            )
            st.session_state["subgroup_modes"] = (
                mode_a,
                mode_b,
                subgroup_bootstrap,
                dataset_fingerprint,
            )
            st.session_state["subgroup_experiment"] = (
                df["experiment_id"].astype(str).unique().tolist()
            )

        except ValueError as exc:
            st.error(f"Subgroup analysis unavailable: {exc}")

    if (
        "subgroup_results" in st.session_state
        and st.session_state.get("subgroup_modes")
        == (mode_a, mode_b, subgroup_bootstrap, dataset_fingerprint)
        and st.session_state.get("subgroup_experiment")
        == df["experiment_id"].astype(str).unique().tolist()
    ):

        subgroup_results = st.session_state["subgroup_results"]

        st.subheader("Category-Level Results")

        display_df = subgroup_results.copy()

        for column in (
            "baseline_asr",
            "defended_asr",
            "improvement",
            "ci_lower",
            "ci_upper",
        ):
            display_df[column] = (
                display_df[column] * 100
            ).round(2)

        display_df = display_df.rename(
            columns={
                "category": "Attack Category",
                "n_pairs": "Completed Pairs",
                "n_clusters": "Source Clusters",
                "baseline_asr": f"{mode_a} ASR (%)",
                "defended_asr": f"{mode_b} ASR (%)",
                "improvement": "Improvement (pp)",
                "ci_lower": "CI Lower (pp)",
                "ci_upper": "CI Upper (pp)",
                "warning": "Data Warning",
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        # --------------------------------------------
        # Forest plot
        # --------------------------------------------

        st.subheader("Subgroup Improvement Forest Plot")

        plot_df = subgroup_results.sort_values(
            "improvement"
        ).reset_index(drop=True)

        fig, ax = plt.subplots(
            figsize=(10, max(4, len(plot_df) * 0.7))
        )

        positions = range(len(plot_df))

        for index, row in plot_df.iterrows():

            improvement = row["improvement"] * 100

            if (
                pd.notna(row["ci_lower"])
                and pd.notna(row["ci_upper"])
            ):
                lower_error = max(
                    0,
                    improvement - row["ci_lower"] * 100,
                )
                upper_error = max(
                    0,
                    row["ci_upper"] * 100 - improvement,
                )

                ax.errorbar(
                    improvement,
                    index,
                    xerr=[
                        [lower_error],
                        [upper_error],
                    ],
                    fmt="o",
                    capsize=4,
                    color="#2563eb",
                )
            else:
                ax.scatter(
                    improvement,
                    index,
                    color="#2563eb",
                )

        ax.axvline(
            0,
            color="#6b7280",
            linestyle="--",
            linewidth=1,
        )

        ax.set_yticks(list(positions))

        ax.set_yticklabels(
            [
                f"{row['category']} "
                f"(n={row['n_pairs']})"
                for _, row in plot_df.iterrows()
            ]
        )

        ax.set_xlabel(
            "Paired Attack-Success Reduction "
            "(percentage points)"
        )

        ax.set_title(
            "Defense Effectiveness by Attack Category"
        )

        ax.grid(
            axis="x",
            alpha=0.2,
        )

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Intervals are percentile source-cluster "
            "bootstrap intervals within each category. "
            "They are exploratory, are not adjusted for "
            "multiple comparisons, and may be unreliable "
            "with few source clusters."
        )

        # --------------------------------------------
        # Category inspection
        # --------------------------------------------

        st.subheader("Explore an Attack Category")

        selected_category = st.selectbox(
            "Choose a category",
            options=subgroup_results["category"].tolist(),
            key="subgroup_category_selector",
        )

        selected_row = subgroup_results[
            subgroup_results["category"]
            == selected_category
        ].iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            f"{mode_a} ASR",
            f"{selected_row['baseline_asr'] * 100:.1f}%",
        )

        c2.metric(
            f"{mode_b} ASR",
            f"{selected_row['defended_asr'] * 100:.1f}%",
        )

        c3.metric(
            "Paired Reduction",
            f"{selected_row['improvement'] * 100:.1f} pp",
        )

        st.write(
            f"**Completed matched attacks:** "
            f"{selected_row['n_pairs']}"
        )

        st.write(
            f"**Source clusters:** "
            f"{selected_row['n_clusters']}"
        )

        if selected_row["warning"]:
            st.warning(selected_row["warning"])

        # --------------------------------------------
        # CSV export
        # --------------------------------------------

        st.download_button(
            "Download Subgroup Results (CSV)",
            data=subgroup_results.to_csv(index=False),
            file_name="evalscope_subgroup_analysis.csv",
            mime="text/csv",
        )

else:
    st.info(
        "Upload a dataset with at least two configurations "
        "to run subgroup analysis."
    )


# ============================================================
# PREDICTIVE MODELING AND CALIBRATION
# ============================================================

from evalscope.predictive_model import evaluate_predictive_models
from evalscope.model_diagnostics import (
    plot_calibration,
    plot_probability_distributions,
)

st.divider()
st.header("Predictive Modeling & Calibration")
st.write(
    "Compare out-of-fold attack-success predictions using "
    "source-cluster-separated cross-validation."
)
st.caption(
    "Only pre-execution category and configuration are used as "
    "predictors. This is an exploratory within-benchmark analysis, "
    "not a test of generalization to unseen attack types or defenses."
)

ml_folds = st.slider(
    "Group-aware cross-validation folds",
    min_value=2,
    max_value=5,
    value=5,
    key="ml_folds",
)
ml_bins = st.slider(
    "Calibration bins",
    min_value=3,
    max_value=15,
    value=8,
    key="ml_bins",
)
ml_key = (dataset_fingerprint, ml_folds)

if st.button("Run Predictive Model Comparison", key="run_ml"):
    try:
        with st.spinner("Training and evaluating models across folds..."):
            ml_results = evaluate_predictive_models(
                df,
                n_splits=ml_folds,
                seed=42,
            )
        st.session_state["ml_results"] = ml_results
        st.session_state["ml_key"] = ml_key
    except (ValueError, RuntimeError) as exc:
        st.session_state.pop("ml_results", None)
        st.session_state.pop("ml_key", None)
        st.error(f"Predictive modeling unavailable: {exc}")

if (
    "ml_results" in st.session_state
    and st.session_state.get("ml_key") == ml_key
):
    ml_results = st.session_state["ml_results"]
    metrics = ml_results["metrics"]
    predictions = ml_results["predictions"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Completed records", ml_results["n_records"])
    c2.metric("Source clusters", ml_results["n_clusters"])
    c3.metric("CV folds", ml_results["n_splits"])

    st.subheader("Out-of-Fold Model Comparison")
    st.dataframe(
        metrics.round(4),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Lower Brier score and log loss are better. "
        "Higher ROC-AUC and average precision are better. "
        "Metrics are pooled across held-out folds."
    )

    st.subheader("Brier Score by Model")
    st.bar_chart(metrics.set_index("model")["brier_score"])

    st.subheader("Calibration Reliability Diagram")
    calibration_fig = plot_calibration(predictions, n_bins=ml_bins)
    st.pyplot(calibration_fig)
    plt.close(calibration_fig)
    st.caption(
        "Each point represents a nonempty probability bin. "
        "Sparse bins and repeated probabilities can make curves "
        "look more certain than the data support."
    )

    st.subheader("Predicted Probability Distributions")
    distribution_fig = plot_probability_distributions(predictions)
    st.pyplot(distribution_fig)
    plt.close(distribution_fig)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download Model Metrics (CSV)",
            data=metrics.to_csv(index=False),
            file_name="evalscope_model_metrics.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Download Out-of-Fold Predictions (CSV)",
            data=predictions.to_csv(index=False),
            file_name="evalscope_oof_predictions.csv",
            mime="text/csv",
        )


# ============================================================
# AUTOMATED RESEARCH REPORT
# ============================================================

from evalscope.report import generate_report

st.divider()
st.header("Automated Research Report")

st.write(
    "Generate a reproducible Markdown research report "
    "containing statistical findings and limitations."
)

if len(analysis_modes) >= 2:

    if st.button("Generate Research Report", type="primary"):
        try:
            with st.spinner("Generating statistical report..."):
                report_text = generate_report(
                    df,
                    mode_a=mode_a,
                    mode_b=mode_b,
                    n_bootstrap=2000,
                    seed=42,
                )

            st.session_state["research_report"] = report_text
            st.session_state["research_report_key"] = (
                dataset_fingerprint, mode_a, mode_b
            )

        except ValueError as exc:
            st.error(f"Report generation failed: {exc}")

    if (
        "research_report" in st.session_state
        and st.session_state.get("research_report_key")
        == (dataset_fingerprint, mode_a, mode_b)
    ):
        st.download_button(
            "Download Research Report (.md)",
            data=st.session_state["research_report"],
            file_name="evalscope_research_report.md",
            mime="text/markdown",
        )

        with st.expander("Preview Research Report"):
            st.markdown(st.session_state["research_report"])

else:
    st.info(
        "Upload a dataset with at least two configurations "
        "to generate a comparative research report."
    )
