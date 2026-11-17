
"""Run exploratory predictive modeling on AgentSecBench."""

from evalscope.data_loader import load_experiment
from evalscope.predictive_model import evaluate_predictive_models


def main():
    df = load_experiment(
        "data/agentsecbench_experiment.json"
    )

    result = evaluate_predictive_models(
        df,
        n_splits=5,
        seed=42,
    )

    print("\nPREDICTIVE MODEL COMPARISON")
    print("=" * 75)

    print(
        f"Records: {result['n_records']}"
    )

    print(
        f"Source clusters: {result['n_clusters']}"
    )

    print(
        f"Cross-validation folds: {result['n_splits']}"
    )

    print()

    print(
        result["metrics"].round(4).to_string(
            index=False
        )
    )

    print(
        "\nLower Brier score and log loss are better."
    )

    print(
        "Higher ROC-AUC and average precision are better."
    )

    print(
        "\nThese are pooled out-of-fold metrics "
        "from source-cluster-separated validation."
    )


if __name__ == "__main__":
    main()
