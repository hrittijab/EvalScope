
"""Import and validate AgentSecBench experiment results."""

from pathlib import Path
import json

import pandas as pd


REQUIRED_FIELDS = {
    "attack_id",
    "source_attack_id",
    "category",
    "mode",
    "attack_success",
    "status",
    "valid_trial",
}

SUPPORTED_STATUSES = {"completed", "truncated", "timeout"}

COLUMNS = [
    "attack_id",
    "source_attack_id",
    "category",
    "mode",
    "attack_success",
    "status",
    "valid_trial",
    "completed",
    "customer_attempt",
    "email_attempt",
    "customer_executed",
    "email_executed",
    "duration_seconds",
    "steps_used",
    "model",
    "experiment_id",
]


def load_experiment(path: str | Path) -> pd.DataFrame:
    """Load a benchmark experiment into a validated DataFrame."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Experiment JSON must contain a list.")

    if not records:
        raise ValueError("Experiment JSON is empty.")

    cleaned = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be an object.")

        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(
                f"Record {index} missing fields: {sorted(missing)}"
            )

        if record["status"] not in SUPPORTED_STATUSES:
            raise ValueError(
                f"Record {index} has unknown status: {record['status']}"
            )

        for field in ("attack_success", "valid_trial"):
            if not isinstance(record[field], bool):
                raise ValueError(
                    f"Record {index}: {field} must be boolean."
                )

        for field in (
            "attack_id",
            "source_attack_id",
            "category",
            "mode",
        ):
            if not isinstance(record[field], str) or not record[field]:
                raise ValueError(
                    f"Record {index}: {field} must be a nonempty string."
                )

        row = {
            column: record.get(column)
            for column in COLUMNS
            if column != "completed"
        }

        row["completed"] = (
            record["status"] == "completed"
            and record["valid_trial"]
        )

        cleaned.append(row)

    df = pd.DataFrame(cleaned, columns=COLUMNS)

    duplicates = df.duplicated(
        subset=["experiment_id", "attack_id", "mode"],
        keep=False,
    )

    if duplicates.any():
        raise ValueError(
            "Duplicate attack/mode records found in experiment."
        )

    return df


def summarize_experiment(df: pd.DataFrame) -> dict:
    """Return descriptive counts without statistical assumptions."""
    return {
        "total_trials": len(df),
        "unique_attacks": int(df["attack_id"].nunique()),
        "source_attacks": int(df["source_attack_id"].nunique()),
        "modes": df["mode"].value_counts().to_dict(),
        "statuses": df["status"].value_counts().to_dict(),
        "categories": df["category"].value_counts().to_dict(),
        "completed_trials": int(df["completed"].sum()),
        "incomplete_trials": int((~df["completed"]).sum()),
    }


def paired_trials(
    df: pd.DataFrame,
    mode_a: str = "baseline",
    mode_b: str = "defended",
    completed_only: bool = True,
) -> pd.DataFrame:
    """Match two configurations by attack within an experiment."""
    selected = df[df["mode"].isin([mode_a, mode_b])].copy()

    if completed_only:
        selected = selected[selected["completed"]]

    a = selected[selected["mode"] == mode_a]
    b = selected[selected["mode"] == mode_b]

    paired = a.merge(
        b,
        on=["experiment_id", "attack_id"],
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )

    if not paired.empty:
        mismatched = (
            (paired["source_attack_id_a"] !=
             paired["source_attack_id_b"])
            |
            (paired["category_a"] != paired["category_b"])
        )

        if mismatched.any():
            raise ValueError(
                "Paired records have inconsistent attack metadata."
            )

    return paired
