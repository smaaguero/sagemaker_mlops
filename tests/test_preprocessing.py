import sys
import os
import pandas as pd
import numpy as np
import pytest

# Add src to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import preprocess_abalone_data, FEATURE_COLUMNS_NAMES, LABEL_COLUMN

def test_preprocess_abalone_data():
    # Create dummy data
    data = {
        "sex": ["M", "F", "I", "M"] * 25,
        "length": np.random.rand(100),
        "diameter": np.random.rand(100),
        "height": np.random.rand(100),
        "whole_weight": np.random.rand(100),
        "shucked_weight": np.random.rand(100),
        "viscera_weight": np.random.rand(100),
        "shell_weight": np.random.rand(100),
        "rings": np.random.randint(1, 20, 100).astype(float)
    }
    
    df = pd.DataFrame(data)
    
    # Run preprocessing
    train, validation, test = preprocess_abalone_data(df)
    
    # Check splits
    total_len = len(df)
    assert len(train) == int(0.7 * total_len)
    assert len(validation) == int(0.15 * total_len) # 0.85 - 0.7 = 0.15
    # The remaining is test, should be around 0.15 * total_len
    assert len(test) == total_len - len(train) - len(validation)
    
    # Check number of columns
    # 1 target column + 3 one-hot encoded columns (M, F, I) + 7 numeric columns = 11 columns
    # Note: OneHotEncoder handle_unknown='ignore' might produce different number of cols if categories are missing,
    # but here we have all 3.
    expected_cols = 1 + 3 + 7
    assert train.shape[1] == expected_cols
    assert validation.shape[1] == expected_cols
    assert test.shape[1] == expected_cols

if __name__ == "__main__":
    pytest.main()