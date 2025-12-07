import json
import pathlib
import pickle
import tarfile
import joblib
import numpy as np
import pandas as pd
import xgboost
import sys

from sklearn.metrics import mean_squared_error
from utils.logger import get_logger

# Configure logging
logger = get_logger(__name__)


def evaluate_model(model, test_df):
    """
    Evaluates the model using the test dataset.

    Args:
        model: Trained XGBoost model.
        test_df (pd.DataFrame): Test dataset.

    Returns:
        dict: Dictionary containing regression metrics.
    """
    try:
        logger.info("Starting model evaluation...")
        
        if test_df.empty:
            raise ValueError("DataFrame cannot be empty for evaluation.")
            
        y_test = test_df.iloc[:, 0].to_numpy()
        X_test_data = test_df.iloc[:, 1:]
        
        X_test = xgboost.DMatrix(X_test_data.values)
        
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        std = np.std(y_test - predictions)
        
        report_dict = {
            "regression_metrics": {
                "mse": {
                    "value": mse,
                    "standard_deviation": std
                },
            },
        }
        
        logger.info(f"Evaluation metrics: MSE={mse}, STD={std}")
        return report_dict

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise e


def main():
    model_path = "/opt/ml/processing/model/model.tar.gz"
    test_path = "/opt/ml/processing/test/test.csv"
    output_dir = "/opt/ml/processing/evaluation"

    try:
        logger.info(f"Extracting model from {model_path}...")
        with tarfile.open(model_path) as tar:
            tar.extractall(path=".")
        
        logger.info("Loading model...")
        model = pickle.load(open("xgboost-model", "rb"))

        logger.info(f"Reading test data from {test_path}...")
        df = pd.read_csv(test_path, header=None)
        
        report_dict = evaluate_model(model, df)

        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        evaluation_path = f"{output_dir}/evaluation.json"
        logger.info(f"Saving evaluation report to {evaluation_path}...")
        with open(evaluation_path, "w") as f:
            f.write(json.dumps(report_dict))
            
        logger.info("Evaluation completed successfully.")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()