import sys
import os
import pytest
import numpy as np
import pandas as pd
import xgboost
from unittest.mock import MagicMock, patch

# Add src to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from evaluation import evaluate_model

# Fixture for a dummy XGBoost model
@pytest.fixture
def dummy_xgboost_model():
    # Create a mock XGBoost model
    mock_model = MagicMock(spec=xgboost.Booster)
    # Configure the predict method to return a predictable array
    mock_model.predict.return_value = np.array([1.0, 2.0, 3.0, 4.0])
    return mock_model

# Fixture for a dummy test DataFrame
@pytest.fixture
def dummy_test_dataframe():
    # Create a pandas DataFrame that mimics the structure expected by evaluate_model
    # First column is the target (y_test), subsequent columns are features (X_test)
    data = {
        'target': [1.1, 2.2, 2.9, 4.3],
        'feature1': [0.1, 0.2, 0.3, 0.4],
        'feature2': [1.0, 1.1, 1.2, 1.3]
    }
    df = pd.DataFrame(data)
    return df

def test_evaluate_model_basic(dummy_xgboost_model, dummy_test_dataframe):
    """
    Test evaluate_model with a simple, predictable dataset.
    """
    # Expected values for MSE and STD based on dummy_test_dataframe and dummy_xgboost_model's predict return
    # y_test = [1.1, 2.2, 2.9, 4.3]
    # predictions = [1.0, 2.0, 3.0, 4.0]
    # errors = [0.1, 0.2, -0.1, 0.3]
    # squared_errors = [0.01, 0.04, 0.01, 0.09]
    # MSE = (0.01 + 0.04 + 0.01 + 0.09) / 4 = 0.15 / 4 = 0.0375
    # STD of errors (0.1, 0.2, -0.1, 0.3)
    # mean_error = (0.1 + 0.2 - 0.1 + 0.3) / 4 = 0.5 / 4 = 0.125
    # variance = ((0.1-0.125)^2 + (0.2-0.125)^2 + (-0.1-0.125)^2 + (0.3-0.125)^2) / 4
    # variance = ((-0.025)^2 + (0.075)^2 + (-0.225)^2 + (0.175)^2) / 4
    # variance = (0.000625 + 0.005625 + 0.050625 + 0.030625) / 4
    # variance = 0.0875 / 4 = 0.021875
    # STD = sqrt(0.021875) approx 0.14789

    metrics = evaluate_model(dummy_xgboost_model, dummy_test_dataframe)

    assert 'regression_metrics' in metrics
    assert 'mse' in metrics['regression_metrics']
    assert 'value' in metrics['regression_metrics']['mse']
    assert 'standard_deviation' in metrics['regression_metrics']['mse']

    assert np.isclose(metrics['regression_metrics']['mse']['value'], 0.0375)
    assert metrics['regression_metrics']['mse']['standard_deviation'] == pytest.approx(0.1479019945774904)

def test_evaluate_model_empty_dataframe(dummy_xgboost_model):
    """
    Test evaluate_model with an empty DataFrame (should raise an error or handle gracefully).
    """
    empty_df = pd.DataFrame(columns=['target', 'feature1', 'feature2'])
    with pytest.raises(ValueError, match="DataFrame cannot be empty for evaluation."):
        evaluate_model(dummy_xgboost_model, empty_df)

def test_evaluate_model_single_row_dataframe(dummy_xgboost_model):
    """
    Test evaluate_model with a single-row DataFrame.
    """
    single_row_data = {
        'target': [5.0],
        'feature1': [1.0],
        'feature2': [2.0]
    }
    single_row_df = pd.DataFrame(single_row_data)

    # Mock predict for single row
    dummy_xgboost_model.predict.return_value = np.array([4.5])

    metrics = evaluate_model(dummy_xgboost_model, single_row_df)

    # y_test = [5.0], predictions = [4.5]
    # error = [0.5], squared_error = [0.25]
    # MSE = 0.25 / 1 = 0.25
    # STD of errors (0.5) = 0
    assert np.isclose(metrics['regression_metrics']['mse']['value'], 0.25)
    assert np.isclose(metrics['regression_metrics']['mse']['standard_deviation'], 0.0)

# Need to update evaluation.py to handle empty dataframe gracefully or raise explicit error
# Currently it will likely fail with an IndexError or similar from pandas/numpy
# Let's read evaluation.py to see where to add the check for empty dataframe
