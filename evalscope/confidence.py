
"""Exact binomial confidence intervals for benchmark outcomes."""

from scipy.stats import beta


def exact_binomial_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> dict:
    """Compute a two-sided Clopper-Pearson confidence interval."""
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise ValueError("Successes and trials must be integers.")

    if trials <= 0:
        raise ValueError("Trials must be positive.")

    if not 0 <= successes <= trials:
        raise ValueError("Successes must be between 0 and trials.")

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    alpha = 1 - confidence

    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    )

    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    )

    return {
        "successes": successes,
        "trials": trials,
        "rate": successes / trials,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "method": "clopper_pearson",
    }
