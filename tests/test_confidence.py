
"""Tests for exact binomial confidence intervals."""

import pytest

from evalscope.confidence import exact_binomial_interval


def test_zero_successes():
    result = exact_binomial_interval(0, 105)

    assert result["rate"] == 0.0
    assert result["ci_lower"] == 0.0
    assert 0.03 < result["ci_upper"] < 0.04


def test_all_successes():
    result = exact_binomial_interval(20, 20)

    assert result["rate"] == 1.0
    assert result["ci_upper"] == 1.0
    assert result["ci_lower"] < 1.0


def test_mixed_outcomes():
    result = exact_binomial_interval(58, 99)

    assert result["ci_lower"] < result["rate"]
    assert result["ci_upper"] > result["rate"]


def test_invalid_inputs():
    with pytest.raises(ValueError):
        exact_binomial_interval(5, 4)

    with pytest.raises(ValueError):
        exact_binomial_interval(0, 0)

    with pytest.raises(ValueError):
        exact_binomial_interval(1, 10, confidence=1.5)


def test_reproducible():
    assert exact_binomial_interval(0, 105) == exact_binomial_interval(0, 105)
