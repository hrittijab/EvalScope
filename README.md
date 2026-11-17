
# EvalScope

### Statistical Reliability and Robustness Analysis for LLM Benchmarks

EvalScope is an open-source Python framework for analyzing the reliability of large language model (LLM) evaluation results.

It goes beyond reporting benchmark accuracy or attack success rates by quantifying statistical uncertainty, evaluating paired experimental outcomes, examining incomplete trials, and measuring ranking stability.

The framework includes an interactive Streamlit dashboard, automated research reports, and reproducible statistical analyses.

## Research Motivation

LLM benchmark results are often summarized using a single performance number.

However, a reported improvement may depend on:

- Which evaluation cases were sampled
- Whether repeated test cases are statistically independent
- How incomplete or timed-out evaluations are handled
- Whether the same configuration consistently ranks best under resampling

EvalScope investigates these questions through statistical inference and sensitivity analysis.

## Features

| Feature | Description |
|---|---|
| Experiment validation | Validates benchmark JSON records and matched evaluation cases |
| Bootstrap confidence intervals | Estimates uncertainty through resampling |
| Cluster bootstrap | Resamples related attacks by source seed |
| Paired comparison | Compares configurations on identical attack cases |
| Exact McNemar test | Tests discordant paired binary outcomes under independence assumptions |
| Exact binomial intervals | Computes Clopper-Pearson confidence intervals |
| Missing-outcome sensitivity | Evaluates best- and worst-case incomplete-trial assumptions |
| Ranking stability | Measures how frequently configurations rank best under resampling |
| Interactive dashboard | Explores datasets, statistical results, and comparisons |
| Automated reports | Generates Markdown research reports |

## Architecture

```text
Benchmark JSON
      |
      v
Data Validation and Pairing
      |
      +----------------------+
      |                      |
      v                      v
Statistical Analysis     Sensitivity Analysis
      |                      |
      +-----------+----------+
                  |
                  v
          Ranking Stability
                  |
          +-------+-------+
          |               |
          v               v
    Streamlit UI     Markdown Reports
```

## Experimental Case Study: AgentSecBench

EvalScope was applied to AgentSecBench, an AI-agent security evaluation framework, to compare a baseline agent configuration with a deterministic authorization-policy defense.

The evaluated experiment used the same underlying LLM across configurations.

### Dataset

| Metric | Value |
|---|---:|
| Evaluation records | 240 |
| Unique generated attacks | 120 |
| Source attack seeds | 30 |
| Completed evaluations | 204 |
| Fully completed attack pairs | 94 |
| Incomplete attack pairs | 26 |

### Attack Success Rates

| Configuration | Successful / Completed | Observed ASR | Exact 95% CI |
|---|---:|---:|---:|
| Baseline | 58/99 | 58.6% | 48.2–68.4% |
| Authorization defense | 0/105 | 0.0% | 0.0–3.5% |

The exact binomial intervals assume independent trials. Generated attacks share source seeds, so these intervals should not be interpreted as cluster-adjusted confidence guarantees.

### Paired Comparison

Among 94 fully completed matched attack pairs:

- Baseline attack success rate: 61.7%
- Defended attack success rate: 0.0%
- Observed reduction: 61.7 percentage points

The analysis also includes ordinary paired bootstrap and seed-cluster bootstrap confidence intervals.

### Incomplete-Trial Sensitivity

Across all 120 matched pairs:

- Worst-case improvement: 35.8 percentage points
- Best-case improvement: 65.8 percentage points

These are logical sensitivity bounds under extreme missing-outcome assignments, not confidence intervals.

### Ranking Stability

Across 5,000 seed-cluster bootstrap resamples of the 94 completed pairs:

- Authorization defense ranked best in 100% of resamples.
- Baseline ranked best in 0% of resamples.

These are empirical bootstrap ranking frequencies, not probabilities of future superiority.

## Technology Stack

- Python
- NumPy
- pandas
- SciPy
- Matplotlib
- Streamlit
- pytest

## Installation

Requires Python 3.10 or newer.

Clone the repository:

```bash
git clone https://github.com/hrittijab/EvalScope.git
cd EvalScope
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Synthetic Demo

Generate a reproducible synthetic dataset:

```bash
python scripts/generate_demo.py
```

This creates:

```text
data/demo_experiment.json
```

The demo contains:

- 120 synthetic attack cases
- 30 source attack seeds
- Four simulated defense configurations
- 480 evaluation records
- Simulated incomplete evaluations

**Important:** Synthetic demo outcomes are illustrative and are not real LLM evaluation results.

## Launch the Dashboard

```bash
python -m streamlit run app.py
```

Open the local Streamlit URL displayed in your terminal.

Upload `data/demo_experiment.json` to explore the statistical analyses.

## Run Tests

```bash
python -m pytest -q
```

## Project Structure

```text
EvalScope/
├── app.py
├── evalscope/
│   ├── data_loader.py
│   ├── statistics.py
│   ├── paired_analysis.py
│   ├── confidence.py
│   ├── sensitivity.py
│   ├── ranking.py
│   ├── visualize.py
│   └── report.py
├── scripts/
│   └── generate_demo.py
├── tests/
├── data/
│   └── demo_experiment.json
├── docs/
├── requirements.txt
└── README.md
```

## Methodological Limitations

1. **Attack dependence:** Generated attacks sharing source seeds may be correlated. Cluster bootstrap partially addresses this dependence but does not guarantee valid inference for all attack distributions.

2. **Incomplete evaluations:** Timeouts and truncated runs may be systematically related to attack difficulty or defense behavior.

3. **Limited generalization:** Findings from one benchmark, model, and evaluation environment cannot establish universal defense effectiveness.

4. **Ranking uncertainty:** Bootstrap ranking frequencies describe stability under resampling of observed cases rather than calibrated probabilities of future performance.

5. **Statistical assumptions:** Exact binomial intervals and conventional McNemar tests assume independence conditions that may not hold for generated attack variants.

## Future Work

- Compare additional language models and defense configurations
- Add benchmark ranking stability visualizations
- Support additional benchmark data formats
- Expand sensitivity analysis for different missingness mechanisms
- Add automated experiment metadata and reproducibility manifests

## Project Relationship

**AgentSecBench** generates AI-agent security evaluation data.

**EvalScope** independently analyzes benchmark reliability, uncertainty, and statistical robustness.

The projects address different stages of the experimental workflow.

## Author

[GitHub](https://github.com/hrittijab)
```