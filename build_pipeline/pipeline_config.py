import os
import yaml
from pathlib import Path

class PipelineConfig:
    """
    Encapsulates pipeline configuration loading and access.
    """
    def __init__(self, config_path=None):
        if config_path is None:
            # Default to pipeline_config.yaml in the project root
            # Assuming this file is in build_pipeline/pipeline_config.py
            self.config_path = Path(__file__).resolve().parent.parent / "pipeline_config.yaml"
        else:
            self.config_path = Path(config_path)
            
        self._config = self._load_config()

    def _load_config(self):
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    @property
    def pipeline_name(self):
        return self._config["pipeline"]["name"]
    
    @property
    def role_arn(self):
        env_var = self._config["pipeline"]["role_arn_env_var"]
        role = os.getenv(env_var)
        if not role:
            raise ValueError(f"Environment variable '{env_var}' for SageMaker Role ARN is not set.")
        return role

    @property
    def bucket_name(self):
        env_var = self._config["pipeline"]["bucket_env_var"]
        bucket = os.getenv(env_var)
        if not bucket:
            raise ValueError(f"Environment variable '{env_var}' for S3 Bucket is not set.")
        return bucket

    @property
    def input_data_uri_suffix(self):
        return self._config["pipeline"]["data"]["input_uri_suffix"]

    @property
    def batch_data_uri_suffix(self):
        return self._config["pipeline"]["data"]["batch_uri_suffix"]
        
    @property
    def processing_config(self):
        return self._config["pipeline"]["processing"]

    @property
    def training_config(self):
        return self._config["pipeline"]["training"]

    @property
    def evaluation_config(self):
        return self._config["pipeline"]["evaluation"]

    @property
    def inference_config(self):
        return self._config["pipeline"]["inference"]
    
    @property
    def model_package_group_name(self):
        return self._config["pipeline"]["inference"]["register"]["model_package_group_name"]

    @property
    def approval_status_default(self):
        return self._config["pipeline"]["inference"]["register"]["approval_status_default"]
    
    @property
    def threshold_mse(self):
        return self._config["pipeline"]["evaluation"]["threshold_mse"]

