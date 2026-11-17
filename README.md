# EvalScope


**Statistical Evaluation, Reliability Analysis, and Robustness Testing for AI Systems**

EvalScope is a Python-based evaluation framework for measuring the
reliability of AI systems through statistical inference, uncertainty
quantification, robustness analysis, and predictive modeling.

Instead of reporting a single accuracy or attack-success score,
EvalScope investigates whether observed improvements are statistically
credible, how sensitive conclusions are to evaluation design, and
whether model confidence reflects actual correctness.

### Evaluation Tracks

| Track | Focus | Key Methods |
|---|---|---|
| AI-Agent Security | Evaluate attack success and defense effectiveness | Paired comparisons, cluster bootstrap, sensitivity analysis |
| Defense Comparison | Compare six security configurations | McNemar tests, Holm correction, security–utility analysis |
| NLP Reliability | Evaluate sentiment classification reliability | Confidence intervals, calibration, selective prediction |

### Project Highlights

- **82 automated tests** covering statistical calculations and evaluation workflows.
- **Three evaluation tracks** spanning AI-agent security and NLP classification.
- **Uncertainty-aware analysis** using confidence intervals, bootstrap resampling, and multiple-testing correction.
- **Predictive modeling** with group-aware cross-validation and probability calibration.
- **Interactive Streamlit dashboard** for exploring evaluation results.
- **Reproducible benchmark artifacts** with documented statistical limitations.
```

Keep your existing `## Key Finding` and AgentSecBench sections. We'll reorganize them later if necessary.

### Step 2 — Update the repository layout

In your existing `## Repository Layout` code block, add these entries under `evalscope/`:

```text
six_defense_loader.py
six_defense_statistics.py
six_defense_dashboard.py
sst2_analysis.py
sst2_calibration.py
sst2_selective.py
sst2_dashboard.py
```

Under `data/`, add:

```text
six_defense_comparison.json
benign_six_defense_comparison.json
sst2_predictions.csv
```

Under `tests/`, make sure `test_six_defense.py` and `test_sst2.py` are represented.

### Step 3 — Verify

Run:

```powershell
git diff --check
git status --short
```

If `git diff --check` produces no output, the formatting is clean.

Once you've made these changes, tell me **done**. Our next step will be a final reproducibility review and then pushing EvalScope to GitHub.

## Key Finding

In an AgentSecBench case study involving 120 matched attacks from 30 source attack seeds, a deterministic authorization-policy defense reduced observed attack success from **61.7% to 0.0%** across 94 completed matched pairs.

The estimated reduction was 61.7 percentage points (source-cluster bootstrap 95% CI: approximately 56.9–65.9 percentage points). Accounting for 26 incomplete pairs, extreme missing-outcome assumptions produced improvement bounds of 35.8–65.8 percentage points.

**The defense also preserved 100% benign-task success (15/15), matching the baseline (15/15), with no observed utility loss on the tested legitimate tasks.**

These results support the effectiveness of the authorization policy within the evaluated benchmark. Generalization remains limited by the single underlying LLM, 30 source attack seeds, and a small benign-task suite.

### Statistical Reliability, Robustness, and Predictive Modeling for LLM Benchmarks

EvalScope is a Python framework for evaluating how reliable conclusions from large language model (LLM) benchmarks are. Rather than relying on a single accuracy or attack-success number, it examines uncertainty, paired outcomes, incomplete evaluations, ranking stability, subgroup differences, and predictive-model calibration.

It includes an interactive Streamlit dashboard, automated Markdown research reports, reproducible analysis scripts, and automated tests.

## Research Questions

- How much does an apparent improvement depend on the sampled benchmark cases?
- What changes when related attacks share a common source seed?
- How sensitive are conclusions to incomplete or timed-out evaluations?
- Do configurations retain their ranking under resampling?
- Can simple pre-execution metadata predict outcomes, and are those probability estimates calibrated?

## Capabilities

| Area | Methods |
|---|---|
| Data validation | Experiment JSON ingestion, completion filtering, matched-attack pairing |
| Statistical inference | Exact binomial intervals, paired bootstrap, source-cluster bootstrap, exact McNemar test (with independence caveat) |
| Missing outcomes | Worst- and best-case sensitivity bounds |
| Ranking | Bootstrap ranking stability and pairwise comparisons |
| Robustness | Leave-one-source-cluster-out (LOCO) analysis |
| Subgroups | Category-level paired effects and cluster-bootstrap intervals |
| Predictive modeling | Constant baseline, logistic regression, random forest |
| Model validation | Five-fold stratified group cross-validation by source attack ID |
| Model diagnostics | ROC-AUC, average precision, Brier score, log loss, calibration plots, probability distributions |
| Outputs | Streamlit dashboard, Markdown research reports, CSV exports, plots, reproducibility manifest |

## Architecture

```text
Benchmark JSON
     |
     v
Validation, Completion Filtering, Matched Pairing
     |
     +--------------------+---------------------+
     |                    |                     |
     v                    v                     v
Statistical          Robustness &         Predictive Modeling
Inference            Sensitivity          (group-aware CV)
     |                    |                     |
     |                    |                     v
     |                    |               Calibration & Metrics
     +--------------------+---------------------+
                          |
                          v
               Streamlit Dashboard
               Research Reports / CSV / Plots
```

## Experimental Case Study: AgentSecBench

EvalScope was applied to [AgentSecBench](https://github.com/hrittijab/AgentSecBench), an AI-agent security evaluation project, to compare a baseline configuration with a deterministic authorization-policy defense. Both configurations used the same underlying language model; the comparison does **not** represent a head-to-head evaluation of different LLMs.

### Dataset and observed outcomes

| Metric | Value |
|---|---:|
| Evaluation records | 240 |
| Unique generated attacks | 120 |
| Source attack seeds | 30 |
| Completed evaluations | 204 |
| Fully completed matched pairs | 94 |
| Incomplete matched pairs | 26 |

| Configuration | Successful / completed | Observed attack success rate | Exact 95% CI |
|---|---:|---:|---:|
| Baseline | 58 / 99 | 58.6% | 48.2–68.4% |
| Authorization defense | 0 / 105 | 0.0% | 0.0–3.5% |

Among the **94 fully completed matched pairs**, the observed attack-success reduction was **61.7 percentage points** (baseline 61.7%; defended 0.0%). Source-cluster bootstrap analysis accounts partially for dependence between attacks derived from the same seed.

Across all 120 matched pairs, the incomplete-trial sensitivity analysis bounded the possible reduction between **35.8 and 65.8 percentage points** under extreme missing-outcome assignments. These are logical bounds, not confidence intervals.

In 5,000 source-cluster bootstrap resamples, the authorization defense ranked best in **100%** of resamples. This is a ranking-stability statistic for the observed benchmark, not a calibrated probability of future superiority.

### Predictive modeling results

Three classifiers were evaluated on completed, valid records using **five-fold stratified group cross-validation**. All records from a source attack seed were assigned to the same fold. Features were restricted to **attack category** and **evaluation configuration**.

| Model | ROC-AUC ↑ | Average precision ↑ | Brier score ↓ | Log loss ↓ |
|---|---:|---:|---:|---:|
| Constant baseline | 0.4828 | 0.2767 | 0.2036 | 0.5972 |
| Logistic regression | 0.9946 | 0.9819 | 0.0178 | 0.1055 |
| Random forest | 0.9946 | 0.9819 | **0.0097** | **0.0427** |

**Interpretation:** The models discriminate outcomes very well *within this benchmark*. Strong results are partly explained by pronounced outcome differences across attack categories and configurations. These models do not process attack text, and the experiment does not demonstrate transfer to new attack families, new LLMs, or unseen defense designs. Probability calibration is also assessed using out-of-fold calibration and distribution plots.




## Extended Case Study: Six-Defense Security Evaluation

EvalScope also evaluates six AI-agent security configurations across
matched attack scenarios and benign tasks.

The experiment contains:

- 30 attack scenarios per defense (180 evaluations)
- 50 benign tasks per defense (300 evaluations)
- Six defense configurations: baseline, authorization,
  prompt_guard, layered, ml_only, and ml_layered

### Observed Results

| Defense | Attack success | Benign-task success |
|---|---:|---:|
| Baseline | 7/30 (23.3%) | 50/50 (100%) |
| Authorization | 0/30 (0%) | 50/50 (100%) |
| Prompt guard | 1/30 (3.3%) | 49/50 (98%) |
| Layered | 0/30 (0%) | 49/50 (98%) |
| ML only | 1/30 (3.3%) | 50/50 (100%) |
| ML layered | 0/30 (0%) | 48/50 (96%) |

### Statistical Methods

EvalScope uses matched-scenario comparisons to estimate differences
in attack-success rates.

- Exact McNemar tests assess discordant paired outcomes.
- Paired bootstrap intervals quantify uncertainty in observed differences.
- Holm correction controls family-wise error across five
  baseline-versus-defense comparisons.
- Security–utility plots illustrate the tradeoff between blocking
  attacks and preserving legitimate task completion.

All five attack comparisons had Holm-adjusted p-values of 0.078125.
Therefore, none reached the 0.05 significance threshold after
multiple-comparison correction.

The results are descriptive evidence from a small benchmark,
not proof that one defense is universally superior.

## Independent NLP Case Study: SST-2 Model Reliability

EvalScope includes an independent sentiment-classification reliability
analysis using 872 precomputed predictions matched to the official
SST-2 validation examples.

The public prediction source documents a DistilBERT SST-2 pipeline,
although the exact checkpoint provenance of the saved predictions
has not been independently verified.

### Classification Performance

| Metric | Result |
|---|---:|
| Validation examples | 872 |
| Correct predictions | 794 |
| Accuracy | 91.06% |
| Exact binomial 95% CI | 88.96%–92.87% |
| Brier score | 0.0834 |
| Log loss | 0.3902 |
| Expected calibration error (10 bins) | 7.84% |
| Mean prediction confidence | 98.34% |

### Selective Prediction

EvalScope evaluates confidence-based abstention to measure the
tradeoff between classification coverage and error rate.

| Confidence threshold | Coverage | Error rate |
|---|---:|---:|
| 50% | 100% | 8.94% |
| 90% | 95.41% | 7.69% |
| 95% | 92.43% | 6.33% |
| 99% | 84.40% | 4.48% |
| 99.5% | 79.47% | 3.03% |

At a 99.5% confidence threshold, the observed error rate among
accepted predictions falls from 8.94% to 3.03%, while approximately
20.5% of examples are rejected.

This illustrates why classification accuracy alone is insufficient
for evaluating model reliability in deployment settings.

### Limitations

- The analysis uses one validation dataset and one set of
  precomputed model predictions.
- Calibration estimates depend on the binning strategy.
- Confidence-based abstention is evaluated retrospectively;
  the thresholds were not validated on an independent test set.
- The analysis does not establish robustness under distribution shift.

### Reproducibility

The repository includes:

- `data/six_defense_comparison.json`
- `data/benign_six_defense_comparison.json`
- `data/sst2_predictions.csv`
- `evalscope/six_defense_dashboard.py`
- `evalscope/sst2_dashboard.py`
- `tests/test_six_defense.py`
- `tests/test_sst2.py`

These files allow the extended analyses to be reproduced without
running additional LLM inference.

The complete automated test suite currently contains 82 passing tests.


## Dashboard Preview

### Experiment Overview

![Experiment Overview](docs/screenshots/ss1.png)

### Statistical Analysis and Defense Effectiveness

![Subgroup Effectiveness Analysis](docs/screenshots/ss2.png)

### Predictive Model Comparison

![Out-of-Fold Model Comparison](docs/screenshots/ss3.png)

### Model Calibration

![Out-of-Fold Model Calibration](docs/screenshots/ss4.png)

### Predicted Probability Distributions

![Prediction Distributions](docs/screenshots/ss5.png)


## Technology Stack

Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib, Streamlit, pytest.

## Quick Start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/hrittijab/EvalScope.git
cd EvalScope
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Run the dashboard

```bash
python -m streamlit run app.py
```

Upload a compatible experiment JSON through the sidebar. If `data/agentsecbench_experiment.json` exists locally, the dashboard can load it as the default dataset.

### Generate synthetic demonstration data

```bash
python -m scripts.generate_demo
```

The synthetic dataset (`data/demo_experiment.json`) contains 120 simulated attack cases, 30 source seeds, four configurations, and 480 evaluation records, including incomplete trials. **Synthetic results are illustrative, not evidence of real model or defense performance.**

### Reproduce statistical analysis

```bash
python scripts/reproduce.py
```

The default demonstration writes statistical outputs to `results/demo/`. The manifest records analysis settings and a SHA-256 dataset hash.

For a compatible custom dataset, consult the script's available CLI arguments; for the AgentSecBench case study, the workflow uses `baseline` and `defended` as the two configuration labels.

### Verify the end-to-end workflow

```bash
python -m scripts.verify_project
```

This verification script runs predictive evaluation, generates diagnostic plots, and produces a research report in `results/verification/` (provided the script and a compatible dataset are present).

### Run tests

```bash
python -m pytest -q
```

## Repository Layout

```text
EvalScope/
├── app.py                         # Streamlit dashboard
├── evalscope/
│   ├── data_loader.py
│   ├── statistics.py
│   ├── confidence.py
│   ├── paired_analysis.py
│   ├── sensitivity.py
│   ├── ranking.py
│   ├── cluster_diagnostics.py
│   ├── subgroup_analysis.py
│   ├── predictive_model.py
│   ├── model_diagnostics.py
│   ├── visualize.py
│   └── report.py
├── scripts/
│   ├── generate_demo.py
│   ├── reproduce.py
│   ├── inspect_predictions.py
│   ├── inspect_calibration.py
│   └── verify_project.py
├── tests/
├── data/
│   └── demo_experiment.json
├── results/
├── requirements.txt
└── README.md
```

Some paths depend on which optional scripts and generated artifacts have been created locally. The AgentSecBench case-study dataset may be excluded from version control; the synthetic demonstration supports public reproduction of the workflow, not independent reproduction of the real-data findings.

## Methodological Limitations

1. **Correlated attacks:** Generated variants sharing source seeds are not necessarily independent. Cluster resampling mitigates, but does not eliminate, this concern.
2. **Incomplete outcomes:** Timeouts and truncated evaluations may be missing non-randomly. Sensitivity bounds expose the range of possible conclusions under extreme assignments.
3. **Finite benchmark coverage:** One model, benchmark, and deterministic defense policy cannot establish universal defense effectiveness.
4. **Limited source clusters:** The case study has 30 source seeds, restricting precision and external validity.
5. **Predictive generalization:** Group-aware cross-validation separates observed source seeds, but does not test transfer to unseen attack categories, models, or policies.
6. **Statistical assumptions:** Exact binomial intervals and conventional McNemar tests rely on independence assumptions that may not hold for related attack variants.
7. **Ranking interpretation:** Bootstrap best-rank frequencies measure resampling stability, not the probability of future superiority.

## Future Work

- Validate on additional models, benchmarks, and defense configurations.
- Add evaluation on entirely held-out attack families.
- Expand missing-data sensitivity models beyond extreme bounds.
- Support additional benchmark schemas and external validation datasets.

## Relationship to AgentSecBench

**AgentSecBench** produces AI-agent security evaluation data. **EvalScope** analyzes the reliability, uncertainty, robustness, and predictability of benchmark outcomes. They are complementary projects covering data generation and statistical evaluation.

## Author

[GitHub — hrittijab](https://github.com/hrittijab)
