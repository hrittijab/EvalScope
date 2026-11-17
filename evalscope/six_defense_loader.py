
"""Load the November AgentSecBench six-defense evaluation."""

import json
from pathlib import Path

import pandas as pd


DEFENSES = (
    "baseline",
    "authorization",
    "prompt_guard",
    "layered",
    "ml_only",
    "ml_layered",
)


def _read_records(path):
    with Path(path).open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("Expected a JSON list.")

    return records


def load_six_defense_attacks(path):
    """Load attack trials into a normalized DataFrame."""
    records = _read_records(path)
    rows = []

    for record in records:
        evaluation = record["evaluation"]
        metadata = record["metadata"]

        rows.append({
            "case_id": record["attack_id"],
            "category": record["category"],
            "defense": record["defense"],
            "success": evaluation["attack_success"],
            "completed": metadata["status"] == "completed",
            "trial": record.get("trial", 1),
            "dataset": "attack",
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("Attack dataset is empty.")

    if not df["defense"].isin(DEFENSES).all():
        raise ValueError("Unknown defense configuration.")

    if df.duplicated(["case_id", "defense", "trial"]).any():
        raise ValueError("Duplicate attack trial detected.")

    return df


def attack_summary(df):
    """Summarize attack success for each defense."""
    completed = df[df["completed"]]

    summary = (
        completed.groupby("defense")["success"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )

    summary.columns = [
        "defense",
        "successful_attacks",
        "total_attacks",
        "attack_success_rate",
    ]

    return summary


def load_six_defense_benign(path):
    """Load benign evaluation trials into a DataFrame."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected a benign benchmark JSON object.")

    records = data.get("results")

    if not isinstance(records, list) or not records:
        raise ValueError("Missing or empty benign results.")

    rows = []

    for record in records:
        evaluation = record["evaluation"]
        defense = record["defense"]
        success = evaluation["benign_success"]

        if defense not in DEFENSES:
            raise ValueError(f"Unknown defense: {defense}")

        if not isinstance(success, bool):
            raise ValueError("benign_success must be boolean.")

        rows.append({
            "case_id": record["case_id"],
            "category": record["category"],
            "defense": defense,
            "success": success,
            "completed": True,
            "trial": 1,
            "dataset": "benign",
        })

    df = pd.DataFrame(rows)

    if df.duplicated(["case_id", "defense", "trial"]).any():
        raise ValueError("Duplicate benign trial detected.")

    return df


def benign_summary(df):
    """Summarize legitimate-task success for each defense."""
    summary = (
        df.groupby("defense")["success"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )

    summary.columns = [
        "defense",
        "successful_tasks",
        "total_tasks",
        "benign_success_rate",
    ]

    return summary


def security_utility_summary(attacks, benign):
    """Combine attack resistance and benign-task utility."""
    attack_stats = attack_summary(attacks)
    benign_stats = benign_summary(benign)

    return attack_stats.merge(
        benign_stats,
        on="defense",
        validate="one_to_one",
    )
