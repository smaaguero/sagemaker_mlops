import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.xgboost.processing import XGBoostProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep, TransformStep, TuningStep
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput, CreateModelInput, TransformInput
from sagemaker.model import Model
from sagemaker.transformer import Transformer
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.model_metrics import MetricsSource, ModelMetrics 
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter, CategoricalParameter
from sagemaker.workflow.pipeline_context import PipelineSession

def get_processing_step(config, pipeline_session, input_data_param):
    """Creates the processing step."""
    
    sklearn_processor = SKLearnProcessor(
        framework_version=config.processing_config["framework_version"],
        instance_type=config.processing_config["instance_type"],
        instance_count=config.processing_config["instance_count"], 
        base_job_name=config.processing_config["base_job_name"],
        sagemaker_session=pipeline_session,
        role=config.role_arn,
    )
    
    processor_args = sklearn_processor.run(
        inputs=[
            ProcessingInput(source=input_data_param, destination="/opt/ml/processing/input"),
            # Inject utils folder to be available for import
            ProcessingInput(source="src/utils", destination="/opt/ml/processing/input/code/utils")
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test")
        ],
        code="src/preprocessing.py",
    ) 

    if isinstance(pipeline_session, PipelineSession):
        return ProcessingStep(
            name="AbaloneProcess",
            step_args=processor_args
        )
    else:
        return sklearn_processor

def get_estimator(config, pipeline_session, region):
    """Creates the XGBoost estimator."""
    default_bucket = pipeline_session.default_bucket()
    model_path = f"s3://{default_bucket}/AbaloneTrain"
    
    image_uri = sagemaker.image_uris.retrieve(
        framework=config.training_config["framework"],
        region=region,
        version=config.training_config["version"],
        py_version=config.training_config["py_version"],
        instance_type=config.training_config["instance_type"]
    )
    
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=config.training_config["instance_type"],
        instance_count=config.training_config["instance_count"],
        output_path=model_path,
        sagemaker_session=pipeline_session,
        role=config.role_arn,
        base_job_name=config.training_config["base_job_name"],
    )
    xgb_train.set_hyperparameters(**config.training_config["hyperparameters"])
    
    return xgb_train, image_uri

def get_training_step(config, pipeline_session, region, process_step):
    """Creates the training step."""
    
    xgb_train, image_uri = get_estimator(config, pipeline_session, region)

    # Note: In local mode, inputs resolution is handled in build_pipeline.py
    # This function is mainly for remote pipeline step creation.
    # If called in local mode, inputs would need to be passed explicitly, 
    # but here we follow the remote pattern.
    
    train_args = xgb_train.fit(
        inputs={
            "train": TrainingInput(
                s3_data=process_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=process_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv"
            )
        },
    )

    if isinstance(pipeline_session, PipelineSession):
        step_train = TrainingStep(
            name="AbaloneTrain",
            step_args=train_args
        )
        return step_train, xgb_train, image_uri
    else:
        return xgb_train, xgb_train, image_uri


def get_tuning_step(config, estimator, inputs):
    """Creates the hyperparameter tuning step."""
    
    hyperparameter_ranges = {
        "eta": ContinuousParameter(
            config.tuning_config["ranges"]["eta"]["min_value"], 
            config.tuning_config["ranges"]["eta"]["max_value"]
        ),
        "max_depth": IntegerParameter(
            config.tuning_config["ranges"]["max_depth"]["min_value"], 
            config.tuning_config["ranges"]["max_depth"]["max_value"]
        ),
        "min_child_weight": IntegerParameter(
            config.tuning_config["ranges"]["min_child_weight"]["min_value"], 
            config.tuning_config["ranges"]["min_child_weight"]["max_value"]
        )
    }

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name=config.tuning_config["objective_metric_name"],
        objective_type=config.tuning_config["objective_type"],
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=config.tuning_config["max_jobs"],
        max_parallel_jobs=config.tuning_config["max_parallel_jobs"],
    )
    
    if isinstance(estimator.sagemaker_session, PipelineSession):
        step_tuning = TuningStep(
            name="AbaloneTuningTest",
            step_args=tuner.fit(inputs=inputs),
        )
        return step_tuning
    else:
        # Local execution
        tuner.fit(inputs=inputs)
        return tuner

def get_evaluation_step(config, pipeline_session, process_step, step_train):
    """Creates the evaluation step."""
    
    xgboost_processor = XGBoostProcessor(
        framework_version=config.training_config["version"],
        py_version=config.training_config["py_version"],
        instance_type=config.evaluation_config["instance_type"],
        instance_count=config.evaluation_config["instance_count"],
        base_job_name=config.evaluation_config["base_job_name"],
        sagemaker_session=pipeline_session,
        role=config.role_arn,
    )
    
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )
    
    # Check if the step_train is a TuningStep or a TrainingStep to get the model artifact
    if isinstance(step_train, TuningStep):
        model_artifact_s3_uri = step_train.get_top_model_s3_uri(
            top_k=0, 
            s3_bucket=pipeline_session.default_bucket(),
            prefix="AbaloneTrain"
        )
    elif isinstance(step_train, HyperparameterTuner):
        # Local mode tuner object
        # In local mode, getting the model artifact path might differ
        # For simplicity, we assume standard structure or user must adapt
        # Typically local jobs output to /tmp or similar, but here we use the s3 logic
         model_artifact_s3_uri = f"s3://{pipeline_session.default_bucket()}/AbaloneTrain/{step_train.best_training_job()}/output/model.tar.gz"

    else:
        model_artifact_s3_uri = step_train.properties.ModelArtifacts.S3ModelArtifacts

    # Handle input data resolution
    if isinstance(pipeline_session, PipelineSession):
        test_data_uri = process_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri
    else:
        # Local mode: process_step is the processor
        test_data_uri = process_step.latest_job.outputs[2].destination

    eval_args = xgboost_processor.run(
        inputs=[
            ProcessingInput(
                source=model_artifact_s3_uri,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=test_data_uri,
                destination="/opt/ml/processing/test"
            ),
            # Inject utils folder to be available for import
            ProcessingInput(source="src/utils", destination="/opt/ml/processing/input/code/utils")
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code="src/evaluation.py",
    )

    if isinstance(pipeline_session, PipelineSession):
        step_eval = ProcessingStep(
            name="AbaloneEval",
            step_args=eval_args,
            property_files=[evaluation_report],
        )
        return step_eval, evaluation_report
    else:
        return xgboost_processor, evaluation_report

def get_create_model_step(config, pipeline_session, image_uri, step_train):
    """Creates the create model step."""
    
    # Check if the step_train is a TuningStep or a TrainingStep to get the model artifact
    if isinstance(step_train, TuningStep):
        model_artifact_s3_uri = step_train.get_top_model_s3_uri(
            top_k=0, 
            s3_bucket=pipeline_session.default_bucket(),
            prefix="AbaloneTrain"
        )
    elif isinstance(step_train, HyperparameterTuner):
         # Local mode
         model_artifact_s3_uri = f"s3://{pipeline_session.default_bucket()}/AbaloneTrain/{step_train.best_training_job()}/output/model.tar.gz"
    else:
        model_artifact_s3_uri = step_train.properties.ModelArtifacts.S3ModelArtifacts

    model = Model(
        image_uri=image_uri,
        model_data=model_artifact_s3_uri,
        sagemaker_session=pipeline_session,
        role=config.role_arn,
    )
    
    inputs = CreateModelInput(
        instance_type=config.inference_config["create_model"]["instance_type"],
        accelerator_type=config.inference_config["create_model"]["accelerator_type"],
    )
    
    if isinstance(pipeline_session, PipelineSession):
        return CreateModelStep(
            name="AbaloneCreateModel",
            model=model,
            inputs=inputs,
        )
    else:
        # Local mode model creation (mostly just object instantiation)
        # We can optionally call create() but in local workflow usually we use transformer directly
        model.create(instance_type=config.inference_config["create_model"]["instance_type"])
        return model

def get_transform_step(config, default_bucket, step_create_model, batch_data_param):
    """Creates the transform step."""
    
    if isinstance(step_create_model, CreateModelStep):
        model_name = step_create_model.properties.ModelName
    else:
        model_name = step_create_model.name

    transformer = Transformer(
        model_name=model_name,
        instance_type=config.inference_config["transform"]["instance_type"],
        instance_count=config.inference_config["transform"]["instance_count"],
        output_path=f"s3://{default_bucket}/AbaloneTransform"
    )
    
    return TransformStep(
            name="AbaloneTransform",
            transformer=transformer,
            inputs=TransformInput(data=batch_data_param)
        )

def get_register_step(config, estimator, step_train, step_eval, model_approval_status_param, pipeline_session):
    """Creates the register model step."""
    
    if isinstance(pipeline_session, PipelineSession):
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri="{}/evaluation.json".format(
                    step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
                ),
                content_type="application/json"
            )
        )
    else:
        # Local mode metrics resolution (approximation)
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri="{}/evaluation.json".format(
                     step_eval.latest_job.outputs[0].destination
                ),
                content_type="application/json"
            )
        )
    
    # Check if the step_train is a TuningStep or a TrainingStep to get the model artifact
    if isinstance(step_train, TuningStep):
        model_artifact_s3_uri = step_train.get_top_model_s3_uri(
            top_k=0, 
            s3_bucket=pipeline_session.default_bucket(),
            prefix="AbaloneTrain"
        )
    elif isinstance(step_train, HyperparameterTuner):
         # Local mode
         model_artifact_s3_uri = f"s3://{pipeline_session.default_bucket()}/AbaloneTrain/{step_train.best_training_job()}/output/model.tar.gz"
    else:
        model_artifact_s3_uri = step_train.properties.ModelArtifacts.S3ModelArtifacts
    
    if isinstance(pipeline_session, PipelineSession):
        return RegisterModel(
            name="AbaloneRegisterModel",
            estimator=estimator,
            model_data=model_artifact_s3_uri,
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=config.inference_config["register"]["inference_instances"],
            transform_instances=config.inference_config["register"]["transform_instances"],
            model_package_group_name=config.model_package_group_name,
            approval_status=model_approval_status_param,
            model_metrics=model_metrics
        )
    else:
        # Register isn't fully supported locally in the same way, usually skipped or mocked.
        # We return estimator as placeholder
        return estimator

def get_condition_step(config, step_eval, evaluation_report, register_step, create_model_step, transform_step):
    """Creates the condition step."""
    
    # Condition step is purely for Pipeline structure. 
    # In local manual mode, we would check the value in python.
    
    if isinstance(step_eval, ProcessingStep):
        # Remote mode
        cond_lte = ConditionLessThanOrEqualTo(
            left=JsonGet(
                step_name=step_eval.name,
                property_file=evaluation_report,
                json_path="regression_metrics.mse.value"
            ),
            right=config.threshold_mse
        )
        
        return ConditionStep(
            name="AbaloneMSECond",
            conditions=[cond_lte],
            if_steps=[register_step, create_model_step, transform_step],
            else_steps=[], 
        )
    else:
        # Local mode: Return None or similar, logic handled in main if needed
        return None
