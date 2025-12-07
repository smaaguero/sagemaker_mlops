import argparse
import os
import requests
import tempfile
import numpy as np
import pandas as pd
import sys

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utils.logger import get_logger

# Configure logging
logger = get_logger(__name__)

# Because this is a headerless CSV file, specify the column names here.
FEATURE_COLUMNS_NAMES = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
LABEL_COLUMN = "rings"

FEATURE_COLUMNS_DTYPE = {
    "sex": str,
    "length": np.float64,
    "diameter": np.float64,
    "height": np.float64,
    "whole_weight": np.float64,
    "shucked_weight": np.float64,
    "viscera_weight": np.float64,
    "shell_weight": np.float64
}
LABEL_COLUMN_DTYPE = {"rings": np.float64}


def merge_two_dicts(x, y):
    """Merges two dictionaries."""
    z = x.copy()
    z.update(y)
    return z


def preprocess_abalone_data(df):
    """
    Applies preprocessing transformations to the Abalone dataset.
    
    Args:
        df (pd.DataFrame): Raw dataframe containing abalone data.
        
    Returns:
        tuple: (train, validation, test) numpy arrays.
    """
    try:
        logger.info("Starting data preprocessing...")
        
        # Separate target
        if LABEL_COLUMN in df.columns:
            y = df.pop(LABEL_COLUMN)
        else:
            raise ValueError(f"Label column '{LABEL_COLUMN}' not found in dataframe.")

        numeric_features = list(FEATURE_COLUMNS_NAMES)
        numeric_features.remove("sex")
        
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        categorical_features = ["sex"]
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        preprocess = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )
        
        logger.info("Fitting and transforming data...")
        X_pre = preprocess.fit_transform(df)
        y_pre = y.to_numpy().reshape(len(y), 1)
        
        X = np.concatenate((y_pre, X_pre), axis=1)
        
        logger.info("Splitting data into train, validation, and test sets...")
        np.random.shuffle(X)
        train, validation, test = np.split(X, [int(.7*len(X)), int(.85*len(X))])
        
        return train, validation, test

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise e


def main():
    base_dir = "/opt/ml/processing"
    input_file = f"{base_dir}/input/abalone-dataset.csv"
    
    try:
        logger.info(f"Reading input data from {input_file}")
        df = pd.read_csv(
            input_file,
            dtype=merge_two_dicts(FEATURE_COLUMNS_DTYPE, LABEL_COLUMN_DTYPE)
        )
        df.columns = FEATURE_COLUMNS_NAMES + [LABEL_COLUMN]

        train, validation, test = preprocess_abalone_data(df)

        logger.info(f"Saving splits to {base_dir}")
        pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
        pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
        pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
        
        logger.info("Preprocessing completed successfully.")

    except FileNotFoundError:
        logger.error(f"Input file not found at {input_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()