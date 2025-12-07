import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline

from pipeline_config import PipelineConfig
from pipeline_steps import (
    get_processing_step,
    get_training_step,
    get_evaluation_step,
    get_create_model_step,
    get_transform_step,
    get_register_step,
    get_condition_step
)

def main():
    # 1. Initialization
    config = PipelineConfig()
    region = boto3.Session().region_name
    sagemaker_session = sagemaker.session.Session()
    default_bucket = sagemaker_session.default_bucket()
    pipeline_session = PipelineSession()
    
    # 2. Define Pipeline Parameters
    input_data_uri = f"s3://{config.bucket_name}/{config.input_data_uri_suffix}"
    batch_data_uri = f"s3://{config.bucket_name}/{config.batch_data_uri_suffix}"

    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount",
        default_value=config.processing_config["instance_count"]
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value=config.approval_status_default
    )
    input_data = ParameterString(
        name="InputData",
        default_value=input_data_uri,
    )
    batch_data = ParameterString(
        name="BatchData",
        default_value=batch_data_uri,
    )

    # 3. Step Creation
    # Processing
    step_process = get_processing_step(config, pipeline_session, input_data)
    
    # Training
    step_train, xgb_estimator, image_uri = get_training_step(config, pipeline_session, region, step_process)
    
    # Evaluation
    step_eval, evaluation_report = get_evaluation_step(config, pipeline_session, step_process, step_train)
    
    # Create Model (for transform)
    step_create_model = get_create_model_step(config, pipeline_session, image_uri, step_train)
    
    # Transform
    step_transform = get_transform_step(config, default_bucket, step_create_model, batch_data)
    
    # Register
    step_register = get_register_step(config, xgb_estimator, step_train, step_eval, model_approval_status)
    
    # Condition
    step_cond = get_condition_step(
        config, 
        step_eval, 
        evaluation_report, 
        step_register, 
        step_create_model, 
        step_transform
    )

    # 4. Pipeline Definition
    pipeline = Pipeline(
        name=config.pipeline_name,
        parameters=[
            processing_instance_count,
            model_approval_status,
            input_data,
            batch_data,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
    )

    # 5. Pipeline Upsert and Execution
    pipeline.upsert(role_arn=config.role_arn)
    
    execution = pipeline.start(
        parameters={
            "InputData": input_data_uri,
            "BatchData": batch_data_uri,
            "ProcessingInstanceCount": config.processing_config["instance_count"],
            "ModelApprovalStatus": config.approval_status_default
        },
        execution_display_name='ejecucion-prueba-refactorizada'
    )
    
    print(f"Pipeline execution started: {execution.arn}")

if __name__ == "__main__":
    main()
