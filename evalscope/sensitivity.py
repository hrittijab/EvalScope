
"""Sensitivity analysis for incomplete benchmark evaluations."""

import pandas as pd

from evalscope.data_loader import paired_trials


def analyze_incomplete_trials(
    df: pd.DataFrame,
    mode_a: str = "baseline",
    mode_b: str = "defended",
) -> dict:
    """Calculate observed rates and worst/best-case paired bounds."""

    pairs = paired_trials(
        df,
        mode_a=mode_a,
        mode_b=mode_b,
        completed_only=False,
    )

    if pairs.empty:
        raise ValueError("No paired attacks found.")

    complete_a = pairs["completed_a"].astype(bool)
    complete_b = pairs["completed_b"].astype(bool)
    both_complete = complete_a & complete_b

    outcomes_a = pairs["attack_success_a"].astype(bool)
    outcomes_b = pairs["attack_success_b"].astype(bool)

    completed_pairs = pairs[both_complete]

    # For incomplete trials, do not assume the recorded
    # attack_success=False represents a confirmed failure.
    best_a = outcomes_a.where(complete_a, True).astype(int)
    best_b = outcomes_b.where(complete_b, False).astype(int)

    worst_a = outcomes_a.where(complete_a, False).astype(int)
    worst_b = outcomes_b.where(complete_b, True).astype(int)

    completed_improvement = (
        (
            completed_pairs["attack_success_a"].astype(int)
            - completed_pairs["attack_success_b"].astype(int)
        ).mean()
        if not completed_pairs.empty
        else None
    )

    return {
        "total_pairs": len(pairs),
        "completed_pairs": int(both_complete.sum()),
        "incomplete_pairs": int((~both_complete).sum()),
        "incomplete_a": int((~complete_a).sum()),
        "incomplete_b": int((~complete_b).sum()),
        "completed_only_improvement": (
            float(completed_improvement)
            if completed_improvement is not None
            else None
        ),
        "best_case_improvement": float((best_a - best_b).mean()),
        "worst_case_improvement": float((worst_a - worst_b).mean()),
        "method": "binary_outcome_sensitivity_bounds",
    }
