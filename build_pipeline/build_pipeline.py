import os
import boto3
import sagemaker
import sagemaker.session
import yaml
from pathlib import Path
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.model import Model
from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.steps import CreateModelStep
from sagemaker.transformer import Transformer
from sagemaker.inputs import TransformInput
from sagemaker.workflow.steps import TransformStep
from sagemaker.model_metrics import MetricsSource, ModelMetrics 
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.properties import PropertyFile

# Load configuration
config_path = Path(__file__).parent.parent / "pipeline_config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Environment variables
MY_BUCKET = os.getenv(config["pipeline"]["bucket_env_var"])
MY_ROLE = os.getenv(config["pipeline"]["role_arn_env_var"])

input_data_uri = f"s3://{MY_BUCKET}/{config['pipeline']['data']['input_uri_suffix']}"
batch_data_uri = f"s3://{MY_BUCKET}/{config['pipeline']['data']['batch_uri_suffix']}"

# 1. Define the session and role
region = boto3.Session().region_name
sagemaker_session = sagemaker.session.Session()
default_bucket = sagemaker_session.default_bucket()
pipeline_session = PipelineSession()
model_package_group_name = config["pipeline"]["inference"]["register"]["model_package_group_name"]

# 2. Define pipeline parameters
processing_instance_count = ParameterInteger(
    name="ProcessingInstanceCount",
    default_value=config["pipeline"]["processing"]["instance_count"]
)
model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value=config["pipeline"]["inference"]["register"]["approval_status_default"]
)
input_data = ParameterString(
    name="InputData",
    default_value=input_data_uri,
)
batch_data = ParameterString(
    name="BatchData",
    default_value=batch_data_uri,
)

# 3. Define the SKLearn processor to preprocess the data
sklearn_processor = SKLearnProcessor(
    framework_version=config["pipeline"]["processing"]["framework_version"],
    instance_type=config["pipeline"]["processing"]["instance_type"],
    instance_count=processing_instance_count,
    base_job_name=config["pipeline"]["processing"]["base_job_name"],
    sagemaker_session=pipeline_session,
    role=MY_ROLE,
)
processor_args = sklearn_processor.run(
    inputs=[
      ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),  
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test")
    ],
    code="src/preprocessing.py",
) 

step_process = ProcessingStep(
    name="AbaloneProcess",
    step_args=processor_args
)

# 4: Define a training step
model_path = f"s3://{default_bucket}/AbaloneTrain"
image_uri = sagemaker.image_uris.retrieve(
    framework=config["pipeline"]["training"]["framework"],
    region=region,
    version=config["pipeline"]["training"]["version"],
    py_version=config["pipeline"]["training"]["py_version"],
    instance_type=config["pipeline"]["training"]["instance_type"]
)
xgb_train = Estimator(
    image_uri=image_uri,
    instance_type=config["pipeline"]["training"]["instance_type"],
    instance_count=config["pipeline"]["training"]["instance_count"],
    output_path=model_path,
    sagemaker_session=pipeline_session,
    role=MY_ROLE,
)
xgb_train.set_hyperparameters(**config["pipeline"]["training"]["hyperparameters"])

train_args = xgb_train.fit(
    inputs={
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "validation": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "validation"
            ].S3Output.S3Uri,
            content_type="text/csv"
        )
    },
)

step_train = TrainingStep(
    name="AbaloneTrain",
    step_args = train_args
)

# 5: Define a processing step for model evaluation
script_eval = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    instance_type=config["pipeline"]["evaluation"]["instance_type"],
    instance_count=config["pipeline"]["evaluation"]["instance_count"],
    base_job_name=config["pipeline"]["evaluation"]["base_job_name"],
    sagemaker_session=pipeline_session,
    role=MY_ROLE,
)
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

eval_args = script_eval.run(
        inputs=[
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=step_process.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
    ],
    code="src/evaluation.py",
)

step_eval = ProcessingStep(
    name="AbaloneEval",
    step_args=eval_args,
    property_files=[evaluation_report],
)

# 6: Define a CreateModelStep for batch transformation
model = Model(
    image_uri=image_uri,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=pipeline_session,
    role=MY_ROLE,
)
inputs = CreateModelInput(
    instance_type=config["pipeline"]["inference"]["create_model"]["instance_type"],
    accelerator_type=config["pipeline"]["inference"]["create_model"]["accelerator_type"],
)
step_create_model = CreateModelStep(
    name="AbaloneCreateModel",
    model=model,
    inputs=inputs,
)

# 7: Define a TransformStep to perform batch transformation
transformer = Transformer(
    model_name=step_create_model.properties.ModelName,
    instance_type=config["pipeline"]["inference"]["transform"]["instance_type"],
    instance_count=config["pipeline"]["inference"]["transform"]["instance_count"],
    output_path=f"s3://{default_bucket}/AbaloneTransform"
)
step_transform = TransformStep(
    name="AbaloneTransform",
    transformer=transformer,
    inputs=TransformInput(data=batch_data)
)

# 8.Define a RegisterModel step to create a model package
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri="{}/evaluation.json".format(
            step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        ),
        content_type="application/json"
    )
)
step_register = RegisterModel(
    name="AbaloneRegisterModel",
    estimator=xgb_train,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=config["pipeline"]["inference"]["register"]["inference_instances"],
    transform_instances=config["pipeline"]["inference"]["register"]["transform_instances"],
    model_package_group_name=model_package_group_name,
    approval_status=model_approval_status,
    model_metrics=model_metrics
)

# 9: Define a condition step to verify model accuracy
cond_lte = ConditionLessThanOrEqualTo(
    left=JsonGet(
        step_name=step_eval.name,
        property_file=evaluation_report,
        json_path="regression_metrics.mse.value"
    ),
    right=config["pipeline"]["evaluation"]["threshold_mse"]
)
step_cond = ConditionStep(
    name="AbaloneMSECond",
    conditions=[cond_lte],
    if_steps=[step_register, step_create_model, step_transform],
    else_steps=[], 
)

# Final step: Define the pipeline
pipeline_name = config["pipeline"]["name"]
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        processing_instance_count,
        model_approval_status,
        input_data,
        batch_data,
    ],
    steps=[step_process, step_train, step_eval, step_cond],
    sagemaker_session=sagemaker_session,
)

if __name__ == "__main__":
    # Crear o actualizar pipeline
    pipeline.upsert(role_arn=MY_ROLE)
    
    # Iniciar ejecución
    execution = pipeline.start(
        parameters={
            "InputData": input_data_uri,
            "BatchData": batch_data_uri,
            "ProcessingInstanceCount": config["pipeline"]["processing"]["instance_count"],
            "ModelApprovalStatus": config["pipeline"]["inference"]["register"]["approval_status_default"]
        },
        execution_display_name = 'ejecucion-prueba-final'
    )