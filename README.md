# Abalone Age Prediction Pipeline (AWS SageMaker)

This project implements an end-to-end Machine Learning (MLOps) pipeline using **AWS SageMaker Pipelines**. It automates the process of data preprocessing, model training, evaluation, and conditional registration for the Abalone dataset.

## Project Overview

The goal of this pipeline is to predict the age of abalone (indicated by the number of rings) based on physical measurements.

The pipeline performs the following steps:
1.  **Preprocessing:** Cleans, scales, and transforms raw data using Scikit-Learn.
2.  **Training:** Trains an XGBoost regression model.
3.  **Evaluation:** Calculates the Mean Squared Error (MSE) of the model on a test set.
4.  **Conditional Check:** Verifies if the model meets the quality threshold (MSE ≤ 6.0).
5.  **Registration & Deployment:** If the check passes, the model is registered in the SageMaker Model Registry, and a Batch Transform job is triggered.

## Project Structure

```text
.
├── build_pipeline/
│   └── build_pipeline.py   # Main script to define and execute the SageMaker Pipeline
├── src/
│   ├── preprocessing.py    # Script for feature engineering (sklearn)
│   └── evaluation.py       # Script for model evaluation (metrics calculation)
├── dataset/                # Local folder for dataset storage (optional)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Prerequisites

Before running this project, ensure you have:

*   **AWS Account** with access to SageMaker and S3.
*   **Python 3.8+** installed locally.
*   **AWS CLI** configured with your credentials (`aws configure`).
*   **S3 Bucket**: A bucket to store input data and model artifacts.
*   **IAM Role**: An execution role with permissions for SageMaker to access S3 and run jobs.

## Installation

1.  Clone this repository:
    ```bash
    git clone <repository-url>
    cd sagemaker_documentacion
    ```

2.  Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The pipeline requires specific environment variables to function. You must set these before running the script.

```bash
export BUCKET="your-sagemaker-bucket-name"
export SAGEMAKER_ROLE_ARN="arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole"
```

*Ensure the input dataset (`abalone-dataset.csv`) is uploaded to your S3 bucket at the path expected by the script (e.g., `s3://{BUCKET}/abalone/input/abalone-dataset.csv`).*

## Usage

To build, upsert, and start the pipeline execution, run:

```bash
python build_pipeline/build_pipeline.py
```

This script will:
1.  Define the pipeline steps (Processing, Training, Evaluation, etc.).
2.  Upload the pipeline definition to AWS SageMaker.
3.  Trigger a new execution named `ejecucion-prueba-final`.

You can monitor the execution progress in the AWS SageMaker Studio UI.

## Pipeline Details

### Preprocessing (`src/preprocessing.py`)
*   **Input:** Raw CSV data.
*   **Logic:**
    *   Numerical features (`length`, `weight`, etc.): Imputed (Median) and Scaled (StandardScaler).
    *   Categorical features (`sex`): One-Hot Encoded.
    *   Splits data into Train (70%), Validation (15%), and Test (15%).
*   **Output:** Headerless CSVs for each split.

### Model Training (`build_pipeline.py`)
*   **Algorithm:** XGBoost (AWS managed container).
*   **Objective:** Linear Regression (`reg:linear`).
*   **Hyperparameters:** Configured in the pipeline script (`num_round=50`, `max_depth=5`, etc.).

### Evaluation (`src/evaluation.py`)
*   Loads the trained model.
*   Predicts on the test set.
*   Calculates MSE and Standard Deviation.
*   Generates an evaluation report (`evaluation.json`).

## License

[MIT License](LICENSE)
