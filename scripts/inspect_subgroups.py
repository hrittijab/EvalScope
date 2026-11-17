
"""Inspect category-level defense effectiveness."""

from evalscope.data_loader import load_experiment
from evalscope.subgroup_analysis import analyze_subgroups


def main():
    df = load_experiment(
        "data/agentsecbench_experiment.json"
    )

    results = analyze_subgroups(
        df,
        mode_a="baseline",
        mode_b="defended",
        n_bootstrap=2000,
        seed=42,
    )

    display = results.copy()

    for column in (
        "baseline_asr",
        "defended_asr",
        "improvement",
        "ci_lower",
        "ci_upper",
    ):
        display[column] = (
            display[column] * 100
        ).round(2)

    print("\nSUBGROUP EFFECTIVENESS ANALYSIS")
    print("=" * 80)
    print(display.to_string(index=False))

    print(
        "\nRates and improvements are expressed "
        "as percentages and percentage points."
    )

    print(
        "Bootstrap intervals resample source-attack "
        "clusters within each subgroup."
    )

    print(
        "Subgroup results are exploratory; "
        "confidence intervals are not adjusted "
        "for multiple comparisons."
    )


if __name__ == "__main__":
    main()
