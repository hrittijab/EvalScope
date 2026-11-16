import numpy as np
import pandas as pd
import scipy
import evalscope


def test_dependencies_import():
    assert np.__version__
    assert pd.__version__
    assert scipy.__version__


def test_project_import():
    assert evalscope is not None


def test_basic_accuracy_calculation():
    outcomes = np.array([1, 1, 0, 1, 0])

    accuracy = outcomes.mean()

    assert accuracy == 0.6