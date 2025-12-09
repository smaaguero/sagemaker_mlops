import boto3
import sagemaker
from sagemaker.local import LocalSession
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.inputs import TrainingInput

from pipeline_config import PipelineConfig
from pipeline_steps import (
    get_processing_step,
    get_estimator,
    get_tuning_step,
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
    
    if config.run_mode == 'local':
        print("Running in LOCAL mode (Sequential Execution)")
        pipeline_session = LocalSession()
        pipeline_session.config = {'local': {'local_code': True}}
        # Override instance types for local execution
        config.processing_config["instance_type"] = "local"
        config.training_config["instance_type"] = "local"
        config.evaluation_config["instance_type"] = "local"
        config.inference_config["create_model"]["instance_type"] = "local"
        config.inference_config["transform"]["instance_type"] = "local"
    else:
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
    
    # Training (Estimator Definition)
    xgb_estimator, image_uri = get_estimator(config, pipeline_session, region)
    
    # Handle Data dependency logic depending on mode
    if config.run_mode == 'local':
        # In local mode with old SDK, steps run eagerly. We need to manually construct inputs from previous job results.
        # step_process is actually the Processor object (after run) or we need to fetch the job.
        # Note: This logic requires pipeline_steps.py to return the object, not the Step, when in local mode.
        
        # This part assumes pipeline_steps.py has been updated to handle non-PipelineSession
        train_input_uri = step_process.latest_job.outputs[0].destination
        validation_input_uri = step_process.latest_job.outputs[1].destination
        test_input_uri = step_process.latest_job.outputs[2].destination
        
        training_inputs = {
            "train": TrainingInput(s3_data=train_input_uri, content_type="text/csv"),
            "validation": TrainingInput(s3_data=validation_input_uri, content_type="text/csv")
        }
    else:
        # Remote mode: Use Property files
        training_inputs = {
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }

    # Tuning (Hyperparameter Optimization)
    step_tuning = get_tuning_step(
        config, 
        xgb_estimator, 
        inputs=training_inputs
    )
    
    # Evaluation (using the best model from tuning)
    step_eval, evaluation_report = get_evaluation_step(config, pipeline_session, step_process, step_tuning)
    
    # Create Model (using the best model from tuning)
    step_create_model = get_create_model_step(config, pipeline_session, image_uri, step_tuning)
    
    # Transform
    step_transform = get_transform_step(config, default_bucket, step_create_model, batch_data)
    
    # Register (using the best model from tuning)
    step_register = get_register_step(config, xgb_estimator, step_tuning, step_eval, model_approval_status, pipeline_session)
    
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
    if config.run_mode != 'local':
        pipeline = Pipeline(
            name=config.pipeline_name,
            parameters=[
                processing_instance_count,
                model_approval_status,
                input_data,
                batch_data,
            ],
            steps=[step_process, step_tuning, step_eval, step_cond],
            sagemaker_session=sagemaker_session,
        )

        # 5. Pipeline Upsert and Execution
        pipeline.upsert(role_arn=config.role_arn)
        
        # execution = pipeline.start(
        #     parameters={
        #         "InputData": input_data_uri,
        #         "BatchData": batch_data_uri,
        #         "ProcessingInstanceCount": config.processing_config["instance_count"],
        #         "ModelApprovalStatus": config.approval_status_default
        #     },
        #     execution_display_name='ejecucion-prueba-tuning'
        # )
        
        # print(f"Pipeline execution started: {execution.arn}")
    else:
        print("Local execution completed (sequential steps).")

if __name__ == "__main__":
    main()
