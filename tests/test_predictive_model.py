
"""Tests for EvalScope predictive modeling."""

import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import StratifiedGroupKFold

from evalscope.predictive_model import (
    FEATURES,
    GROUP,
    TARGET,
    evaluate_predictive_models,
    prepare_prediction_data,
)


@pytest.fixture
def prediction_dataset():
    """Create a deterministic dataset with independent source groups."""

    rows = []

    for cluster in range(12):
        for mode in ("baseline", "defended"):
            for category in (
                "instruction_override",
                "workflow_hijacking",
            ):
                rows.append({
                    "attack_id": f"attack-{cluster}-{category}",
                    "source_attack_id": f"source-{cluster}",
                    "category": category,
                    "mode": mode,
                    "attack_success": (
                        mode == "baseline"
                        and (
                            category == "instruction_override"
                            or cluster % 3 != 0
                        )
                    ),
                    "completed": True,
                    "valid_trial": True,
                })

    return pd.DataFrame(rows)


def test_preparation_retains_valid_trials(prediction_dataset):
    result = prepare_prediction_data(prediction_dataset)

    assert len(result) == 48
    assert result[GROUP].nunique() == 12


def test_preparation_excludes_incomplete_trials(prediction_dataset):
    data = prediction_dataset.copy()
    data.loc[0, "completed"] = False
    data.loc[1, "valid_trial"] = False

    result = prepare_prediction_data(data)

    assert len(result) == 46


def test_preparation_rejects_missing_columns(prediction_dataset):
    data = prediction_dataset.drop(columns=["category"])

    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_prediction_data(data)


def test_preparation_rejects_insufficient_clusters(prediction_dataset):
    data = prediction_dataset[
        prediction_dataset[GROUP].isin(
            ["source-0", "source-1"]
        )
    ]

    with pytest.raises(ValueError, match="five source clusters"):
        prepare_prediction_data(data)


def test_group_folds_do_not_overlap(prediction_dataset):
    data = prepare_prediction_data(prediction_dataset)

    X = data[FEATURES]
    y = data[TARGET].astype(int)
    groups = data[GROUP]

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    all_validation_indices = []

    for train_idx, test_idx in splitter.split(X, y, groups):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])

        assert train_groups.isdisjoint(test_groups)

        all_validation_indices.extend(test_idx.tolist())

    assert sorted(all_validation_indices) == list(range(len(data)))


def test_evaluation_returns_three_models(prediction_dataset):
    result = evaluate_predictive_models(
        prediction_dataset,
        n_splits=5,
        seed=42,
    )

    assert set(result["metrics"]["model"]) == {
        "Constant Baseline",
        "Logistic Regression",
        "Random Forest",
    }

    assert len(result["predictions"]) == 48 * 3


def test_predictions_are_valid_probabilities(prediction_dataset):
    result = evaluate_predictive_models(
        prediction_dataset,
        n_splits=5,
        seed=42,
    )

    probabilities = result["predictions"]["predicted_probability"]

    assert probabilities.between(0, 1).all()
    assert np.isfinite(probabilities).all()


def test_evaluation_is_reproducible(prediction_dataset):
    first = evaluate_predictive_models(
        prediction_dataset,
        n_splits=5,
        seed=42,
    )

    second = evaluate_predictive_models(
        prediction_dataset,
        n_splits=5,
        seed=42,
    )

    pd.testing.assert_frame_equal(
        first["metrics"],
        second["metrics"],
    )

    pd.testing.assert_frame_equal(
        first["predictions"],
        second["predictions"],
    )


def test_invalid_number_of_folds(prediction_dataset):
    with pytest.raises(ValueError, match="at least 2"):
        evaluate_predictive_models(
            prediction_dataset,
            n_splits=1,
        )


def test_too_many_folds(prediction_dataset):
    with pytest.raises(ValueError, match="Not enough source clusters"):
        evaluate_predictive_models(
            prediction_dataset,
            n_splits=20,
        )
