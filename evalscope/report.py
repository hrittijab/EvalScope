
"""Generate a reproducible Markdown benchmark evaluation report."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evalscope.confidence import exact_binomial_interval
from evalscope.data_loader import paired_trials
from evalscope.paired_analysis import compare_defenses
from evalscope.ranking import analyze_ranking_stability
from evalscope.sensitivity import analyze_incomplete_trials


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

    # The analysis modules expect one experiment.
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

## 7. Threats to Validity

### Dependence between attack cases

Generated attacks can share source attack seeds. Treating
all generated variants as independent can understate
uncertainty.

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

## 8. Reproducibility

- Bootstrap resamples: {n_bootstrap}
- Random seed: {seed}
- Paired unit: attack ID within experiment
- Cluster unit: source attack ID
- Completion requirement: completed and valid
- Ranking direction: lower attack success is better

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
