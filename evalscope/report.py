
"""Generate a reproducible Markdown benchmark evaluation report."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evalscope.cluster_diagnostics import leave_one_cluster_out
from evalscope.confidence import exact_binomial_interval
from evalscope.paired_analysis import compare_defenses
from evalscope.ranking import analyze_ranking_stability
from evalscope.sensitivity import analyze_incomplete_trials
from evalscope.subgroup_analysis import analyze_subgroups
from evalscope.predictive_model import evaluate_predictive_models


def generate_report(
    df: pd.DataFrame,
    mode_a: str = "baseline",
    mode_b: str = "defended",
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> str:
    """Return a complete Markdown evaluation report."""

    if mode_a == mode_b:
        raise ValueError("Select two different configurations.")

    experiment_ids = df["experiment_id"].dropna().unique()

    if len(experiment_ids) != 1:
        raise ValueError(
            "Report generation requires exactly one experiment."
        )

    selected = df[df["mode"].isin([mode_a, mode_b])].copy()

    if selected.empty:
        raise ValueError("No matching configurations.")

    rates = {}

    for mode in (mode_a, mode_b):
        completed = selected[
            (selected["mode"] == mode)
            & selected["completed"]
        ]

        if completed.empty:
            raise ValueError(
                f"No completed trials for configuration: {mode}"
            )

        successes = int(completed["attack_success"].sum())

        rates[mode] = exact_binomial_interval(
            successes,
            len(completed),
        )

    ordinary = compare_defenses(
        selected,
        mode_a=mode_a,
        mode_b=mode_b,
        n_bootstrap=n_bootstrap,
        seed=seed,
        cluster=False,
    )

    clustered = compare_defenses(
        selected,
        mode_a=mode_a,
        mode_b=mode_b,
        n_bootstrap=n_bootstrap,
        seed=seed,
        cluster=True,
    )

    sensitivity = analyze_incomplete_trials(
        selected,
        mode_a=mode_a,
        mode_b=mode_b,
    )

    ranking = analyze_ranking_stability(
        selected,
        modes=[mode_a, mode_b],
        n_bootstrap=n_bootstrap,
        seed=seed,
        cluster=True,
    )

    loco = leave_one_cluster_out(
        selected,
        mode_a=mode_a,
        mode_b=mode_b,
    )
    # Predictive modeling is supplementary: small datasets should still
    # produce the statistical research report.
    ml_rows = []
    ml_unavailable = None
    try:
        predictive = evaluate_predictive_models(
            selected,
            n_splits=5,
            seed=seed,
        )
        for _, result in predictive["metrics"].iterrows():
            ml_rows.append(
                f"| {result['model']} "
                f"| {result['roc_auc']:.4f} "
                f"| {result['average_precision']:.4f} "
                f"| {result['brier_score']:.4f} "
                f"| {result['log_loss']:.4f} |"
            )
    except ValueError as exc:
        ml_unavailable = str(exc)

    subgroup_results = None
    if "category" in selected.columns and selected["category"].notna().all():
        subgroup_results = analyze_subgroups(
            selected,
            mode_a=mode_a,
            mode_b=mode_b,
            group_column="category",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

    def pct(value):
        return f"{value * 100:.1f}%"

    def pp(value):
        return f"{value * 100:.1f} pp"

    def interval_text(result):
        return (
            f"{pct(result['ci_lower'])} to "
            f"{pct(result['ci_upper'])}"
        )

    rate_rows = []

    for mode in (mode_a, mode_b):
        r = rates[mode]
        rate_rows.append(
            f"| {mode} | {r['successes']}/{r['trials']} | "
            f"{pct(r['rate'])} | {interval_text(r)} |"
        )

    ranking_rows = []

    for mode in (mode_a, mode_b):
        ranking_rows.append(
            f"| {mode} | "
            f"{pct(ranking['observed_success_rates'][mode])} | "
            f"{pct(ranking['best_rank_probability'][mode])} | "
            f"{ranking['average_rank'][mode]:.2f} |"
        )

    subgroup_section = """## 8. Subgroup Effectiveness Analysis

Subgroup analysis was unavailable because the dataset does not
contain complete attack-category labels.

"""

    if subgroup_results is not None:
        subgroup_rows = []
        for _, subgroup in subgroup_results.iterrows():
            if pd.notna(subgroup["ci_lower"]):
                ci_text = (
                    f"{pp(subgroup['ci_lower'])} to "
                    f"{pp(subgroup['ci_upper'])}"
                )
            else:
                ci_text = "Unavailable"

            warning = subgroup["warning"] or "None"
            subgroup_rows.append(
                f"| {subgroup['category']} "
                f"| {subgroup['n_pairs']} "
                f"| {subgroup['n_clusters']} "
                f"| {pct(subgroup['baseline_asr'])} "
                f"| {pct(subgroup['defended_asr'])} "
                f"| {pp(subgroup['improvement'])} "
                f"| {ci_text} "
                f"| {warning} |"
            )

        subgroup_section = (
            "## 8. Subgroup Effectiveness Analysis\n\n"
            "This analysis compares configurations across attack categories "
            "using completed matched pairs.\n\n"
            "| Attack Category | Pairs | Source Clusters | Reference ASR | "
            "Comparison ASR | Reduction | Cluster Bootstrap 95% CI | Warning |\n"
            "|---|---:|---:|---:|---:|---:|---:|---|\n"
            + "\n".join(subgroup_rows)
            + "\n\nPositive reductions indicate lower attack success under "
            "the comparison configuration.\n\n"
            "Intervals use source-cluster bootstrap resampling within "
            "each subgroup. They are exploratory, are not adjusted for "
            "multiple comparisons, and can be unreliable with few clusters. "
            "Zero-width intervals do not imply certainty about unseen attacks.\n\n"
        )

    generated_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    report = f"""# EvalScope: LLM Benchmark Reliability Report

**Generated:** {generated_at}

**Experiment:** `{experiment_ids[0]}`

**Reference configuration:** `{mode_a}`

**Comparison configuration:** `{mode_b}`

**Bootstrap resamples:** {n_bootstrap}

**Random seed:** {seed}

---

## 1. Executive Summary

This report evaluates the reliability of an AI-agent security
benchmark by comparing two configurations on matched attacks.

Among {ordinary['n_pairs']} fully completed attack pairs,
the observed reduction in attack success was
**{pp(ordinary['improvement'])}** when comparing
`{mode_b}` against `{mode_a}`.

The seed-cluster bootstrap 95% confidence interval for this
paired reduction was **{pp(clustered['ci_lower'])} to
{pp(clustered['ci_upper'])}**.

These findings describe the evaluated benchmark and should
not be interpreted as guarantees about future attacks.

## 2. Dataset and Completion

| Metric | Count |
|---|---:|
| Selected evaluation records | {len(selected)} |
| Unique attack cases | {selected['attack_id'].nunique()} |
| Source attack seeds | {selected['source_attack_id'].nunique()} |
| Fully completed pairs | {sensitivity['completed_pairs']} |
| Incomplete pairs | {sensitivity['incomplete_pairs']} |

Only trials marked completed and valid are included in
the completed-case attack success calculations.

## 3. Attack Success Rates

| Configuration | Successful / Completed | ASR | Exact 95% CI |
|---|---:|---:|---:|
{chr(10).join(rate_rows)}

Exact Clopper-Pearson intervals assume independent Bernoulli
trials. Because multiple generated attacks may share source
seeds, these intervals are illustrative rather than
cluster-adjusted confidence guarantees.

## 4. Paired Defense Comparison

| Metric | Result |
|---|---:|
| Completed matched pairs | {ordinary['n_pairs']} |
| Observed reduction | {pp(ordinary['improvement'])} |
| Ordinary bootstrap 95% CI | {pp(ordinary['ci_lower'])} to {pp(ordinary['ci_upper'])} |
| Seed-cluster bootstrap 95% CI | {pp(clustered['ci_lower'])} to {pp(clustered['ci_upper'])} |

The comparison uses matched attacks that completed
successfully under both configurations.

The ordinary bootstrap resamples individual matched attacks.
The cluster bootstrap resamples source attack seeds to
partially account for related generated attacks.

The exact McNemar test, where reported separately, assumes
independent matched pairs. It should not be treated as
cluster-adjusted evidence.

## 5. Incomplete-Trial Sensitivity

| Scenario | Reduction |
|---|---:|
| Worst-case missing outcomes | {pp(sensitivity['worst_case_improvement'])} |
| Best-case missing outcomes | {pp(sensitivity['best_case_improvement'])} |
| Completed-only estimate | {pp(sensitivity['completed_only_improvement'])} |

The worst-case and best-case endpoints assign hypothetical
outcomes to incomplete evaluations.

These endpoints are **sensitivity bounds, not confidence
intervals**. They use all matched pairs, whereas the
completed-only estimate uses only fully completed pairs.

## 6. Ranking Stability

**Method:** Seed-cluster bootstrap

**Common completed attacks:** {ranking['n_common_attacks']}

**Excluded attacks:** {ranking['n_excluded_attacks']}

**Source clusters:** {ranking['n_source_clusters']}

| Configuration | Observed ASR | Best-rank frequency | Average rank |
|---|---:|---:|---:|
{chr(10).join(ranking_rows)}

Lower attack success rates rank better.

When configurations tie, ranking credit is split equally.

Bootstrap best-rank frequencies measure ranking stability
under resampling of the observed benchmark. They are not
calibrated probabilities of future superiority.

## 7. Leave-One-Cluster-Out Robustness Analysis

This analysis evaluates how sensitive the observed paired
improvement is to excluding individual source-attack seeds.

Each source cluster is excluded once, and the paired
improvement is recalculated using the remaining completed
attack pairs.

| Metric | Result |
|---|---:|
| Completed paired trials | {loco['n_pairs']} |
| Source attack clusters | {loco['n_clusters']} |
| Observed improvement | {pp(loco['observed_improvement'])} |
| Minimum LOCO improvement | {pp(loco['min_loco_improvement'])} |
| Maximum LOCO improvement | {pp(loco['max_loco_improvement'])} |
| Largest absolute change | {pp(loco['max_absolute_change'])} |
| All LOCO estimates positive | {loco['all_loco_positive']} |

### Interpretation

The observed paired reduction was
**{pp(loco['observed_improvement'])}**.

After excluding each source cluster individually, the
estimated reduction ranged from
**{pp(loco['min_loco_improvement'])}** to
**{pp(loco['max_loco_improvement'])}**.

The largest absolute change was
**{pp(loco['max_absolute_change'])}**.

A narrow LOCO range indicates limited sensitivity to
any single observed source-attack seed.

This analysis is a robustness diagnostic. It is not
a confidence interval, a formal hypothesis test, or
evidence of generalization to unseen attacks.

{subgroup_section}## 9. Predictive Machine Learning Evaluation

When supported by the dataset, three models are evaluated using five-fold
stratified group cross-validation:

- Constant-probability baseline
- Logistic Regression
- Random Forest

Only information available before attack execution
was used as predictive features:

- Attack category
- Evaluation configuration

Source attack IDs were used exclusively for
cross-validation grouping.

| Model | ROC-AUC | Average Precision | Brier Score | Log Loss |
|---|---:|---:|---:|---:|
{chr(10).join(ml_rows) if ml_rows else "| Unavailable | — | — | — | — |"}

### Interpretation

Higher ROC-AUC and average precision indicate
better discrimination.

Lower Brier score and log loss indicate better
probability predictions.

{f"Predictive evaluation unavailable: {ml_unavailable}" if ml_unavailable else "Metrics are calculated from pooled out-of-fold predictions, with source attack clusters separated between training and validation folds."}

These findings describe predictive performance
within the observed benchmark.

The experiment does not establish generalization
to new attack categories, unseen language models,
or substantially different defense configurations.


## 10. Threats to Validity

### Dependence between attack cases

Generated attacks can share source attack seeds. Treating
all generated variants as independent can understate
uncertainty.

The seed-cluster bootstrap partially addresses this
dependence. The leave-one-cluster-out analysis additionally
examines whether individual source seeds dominate the
observed effect.

Neither method guarantees generalization to independent,
previously unseen attacks.

### Incomplete evaluations

Timeouts and truncated runs may not be missing at random.
Completed-only analysis may therefore be biased.

### Limited configuration coverage

The analysis compares the selected configurations, not
all possible defenses or language models.

### Benchmark generalization

Results may depend on the attack generator, evaluation
environment, tool permissions, model, and attack distribution.

### Statistical assumptions

Cluster bootstrap intervals depend on the number and
representativeness of source clusters. Small cluster
counts can limit interval reliability.

The exact binomial intervals are not cluster-adjusted.
The conventional McNemar test assumes independent
matched pairs.


## 11. Reproducibility

- Bootstrap resamples: {n_bootstrap}
- Random seed: {seed}
- Paired unit: attack ID within experiment
- Cluster unit: source attack ID
- Completion requirement: completed and valid
- Ranking direction: lower attack success is better
- Robustness diagnostic: leave-one-source-cluster-out
- Predictive validation: five-fold stratified group cross-validation
- Predictive grouping: source attack ID
- Predictive features: attack category and configuration
- Predictive random seed: {seed}

This report was generated automatically by EvalScope.

---

*EvalScope — Statistical Reliability and Robustness
Analysis for LLM Benchmarks.*
"""

    return report


def save_report(
    report: str,
    output_path: str | Path,
) -> Path:
    """Write a Markdown report to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")

    return path
