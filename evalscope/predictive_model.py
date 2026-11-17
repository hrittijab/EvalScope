
"""Leakage-aware predictive modeling for EvalScope."""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURES = ["category", "mode"]
TARGET = "attack_success"
GROUP = "source_attack_id"


def prepare_prediction_data(df):
    """Select completed, valid trials and pre-run features."""

    required = FEATURES + [
        TARGET,
        GROUP,
        "completed",
        "valid_trial",
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    selected = df[
        df["completed"].eq(True)
        & df["valid_trial"].eq(True)
    ].copy()

    selected = selected.dropna(
        subset=FEATURES + [TARGET, GROUP]
    )

    if selected.empty:
        raise ValueError("No usable completed trials.")

    selected[TARGET] = selected[TARGET].astype(int)

    if not selected[TARGET].isin([0, 1]).all():
        raise ValueError("Target must be binary.")

    if selected[GROUP].nunique() < 5:
        raise ValueError(
            "At least five source clusters are required."
        )

    return selected


def make_models():
    """Construct models with categorical preprocessing."""

    def pipeline(classifier):
        return Pipeline([
            (
                "preprocess",
                ColumnTransformer([
                    (
                        "categorical",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),
                        FEATURES,
                    )
                ]),
            ),
            ("model", classifier),
        ])

    return {
        "Constant Baseline": DummyClassifier(
            strategy="prior"
        ),
        "Logistic Regression": pipeline(
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            )
        ),
        "Random Forest": pipeline(
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )
        ),
    }


def _positive_probability(model, X):
    """
    Return probability of class 1.

    Handles training folds containing only one class.
    """

    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    if 1 in classes:
        return probabilities[:, classes.index(1)]

    return np.zeros(len(X), dtype=float)


def evaluate_predictive_models(
    df,
    n_splits=5,
    seed=42,
):
    """
    Evaluate models using stratified group cross-validation.

    Source attack seeds never appear in both training
    and validation within the same fold.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    selected = prepare_prediction_data(df)

    if selected[GROUP].nunique() < n_splits:
        raise ValueError(
            "Not enough source clusters for the requested folds."
        )

    X = selected[FEATURES]
    y = selected[TARGET].to_numpy(dtype=int)
    groups = selected[GROUP]

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    splits = list(splitter.split(X, y, groups))

    models = make_models()

    results = []
    predictions = []

    for model_name, model in models.items():

        oof_probabilities = np.full(
            len(selected),
            np.nan,
            dtype=float,
        )

        fold_ids = np.full(
            len(selected),
            -1,
            dtype=int,
        )

        for fold, (train_idx, test_idx) in enumerate(
            splits, start=1
        ):

            if len(np.unique(y[train_idx])) < 2:
                raise ValueError(
                    "A training fold contains only one "
                    "outcome class. Use fewer folds or "
                    "a dataset with more outcome variation."
                )

            model.fit(
                X.iloc[train_idx],
                y[train_idx],
            )

            probabilities = _positive_probability(
                model,
                X.iloc[test_idx],
            )

            oof_probabilities[test_idx] = probabilities
            fold_ids[test_idx] = fold

        if np.isnan(oof_probabilities).any():
            raise RuntimeError(
                "Some records did not receive predictions."
            )

        metrics = {
            "model": model_name,
            "n_records": len(selected),
            "n_clusters": selected[GROUP].nunique(),
            "brier_score": brier_score_loss(
                y,
                oof_probabilities,
            ),
            "log_loss": log_loss(
                y,
                np.clip(
                    oof_probabilities,
                    1e-12,
                    1 - 1e-12,
                ),
                labels=[0, 1],
            ),
            "roc_auc": (
                roc_auc_score(y, oof_probabilities)
                if len(np.unique(y)) == 2
                else np.nan
            ),
            "average_precision": (
                average_precision_score(
                    y,
                    oof_probabilities,
                )
                if y.sum() > 0
                else np.nan
            ),
        }

        results.append(metrics)

        for index, probability in enumerate(
            oof_probabilities
        ):
            predictions.append({
                "model": model_name,
                "attack_id": selected.iloc[index]["attack_id"],
                "source_attack_id": groups.iloc[index],
                "category": X.iloc[index]["category"],
                "mode": X.iloc[index]["mode"],
                "actual": int(y[index]),
                "predicted_probability": float(
                    probability
                ),
                "fold": int(fold_ids[index]),
            })

    return {
        "metrics": pd.DataFrame(results),
        "predictions": pd.DataFrame(predictions),
        "n_records": len(selected),
        "n_clusters": selected[GROUP].nunique(),
        "n_splits": n_splits,
    }
