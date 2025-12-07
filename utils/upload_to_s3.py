import os
import sys
import yaml
import boto3
from pathlib import Path
from botocore.exceptions import ClientError

# Adjust path for importing from src.utils
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from utils.logger import get_logger

# Configure logging
logger = get_logger(__name__)

def load_config(config_path):
    """Load the pipeline configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        sys.exit(1)

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket."""
    if object_name is None:
        object_name = os.path.basename(file_name)

    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_name, bucket, object_name)
        logger.info(f"Successfully uploaded {file_name} to s3://{bucket}/{object_name}")
    except ClientError as e:
        logger.error(f"Failed to upload {file_name}: {e}")
        return False
    return True

def main():
    # Define paths
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "pipeline_config.yaml"
    dataset_dir = base_dir / "dataset"

    # Check if dataset directory exists
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found at {dataset_dir}")
        sys.exit(1)

    # Load configuration
    config = load_config(config_path)
    
    # Get bucket name
    bucket_env_var = config["pipeline"].get("bucket_env_var")
    bucket_name = os.getenv(bucket_env_var)
    
    if not bucket_name:
        logger.error(f"Environment variable '{bucket_env_var}' not set.")
        logger.info("Please export the bucket name: export BUCKET=<your-bucket-name>")
        sys.exit(1)

    # Get target S3 path from config
    input_uri_suffix = config["pipeline"]["data"]["input_uri_suffix"]
    target_key = input_uri_suffix
    target_prefix = str(Path(target_key).parent)

    files_uploaded = 0
    # Iterate over files in dataset directory
    for file_path in dataset_dir.iterdir():
        if file_path.is_file() and not file_path.name.startswith('.'):
            # Determine S3 key
            # If the file matches the configured input file name, use the exact key
            if file_path.name == Path(target_key).name:
                s3_key = target_key
            else:
                # Otherwise, upload to the same "folder" in S3
                s3_key = f"{target_prefix}/{file_path.name}"
            
            logger.info(f"Uploading {file_path.name}...")
            if upload_file(str(file_path), bucket_name, s3_key):
                files_uploaded += 1

    if files_uploaded == 0:
        logger.warning("No files found in dataset directory to upload.")
    else:
        logger.info(f"Finished. Uploaded {files_uploaded} files.")

if __name__ == "__main__":
    main()

