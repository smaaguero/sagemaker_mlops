import boto3
import sys

def debug_last_pipeline_execution(pipeline_name):
    client = boto3.client("sagemaker")
    
    print(f"Fetching executions for pipeline: {pipeline_name}...")
    
    try:
        response = client.list_pipeline_executions(
            PipelineName=pipeline_name,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1
        )
    except client.exceptions.ResourceNotFound:
        print(f"Pipeline '{pipeline_name}' not found.")
        return

    if not response["PipelineExecutionSummaries"]:
        print("No executions found.")
        return

    latest_execution = response["PipelineExecutionSummaries"][0]
    execution_arn = latest_execution["PipelineExecutionArn"]
    status = latest_execution["PipelineExecutionStatus"]
    
    print(f"\nLatest Execution ARN: {execution_arn}")
    print(f"Status: {status}")

    if status == "Failed":
        desc = client.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
        print(f"Execution Failure Reason: {desc.get('FailureReason', 'No global failure reason provided.')}")

    print("\n--- Step Details ---")
    
    steps_response = client.list_pipeline_execution_steps(
        PipelineExecutionArn=execution_arn,
        SortOrder="Descending" 
    )
    
    for step in steps_response["PipelineExecutionSteps"]:
        step_name = step["StepName"]
        step_status = step["StepStatus"]
        print(f"\nStep: {step_name}")
        print(f"Status: {step_status}")
        
        if step_status == "Failed":
            print(f"Failure Reason: {step.get('FailureReason', 'No failure reason provided.')}")
            # Sometimes there is metadata with more info
            if "Metadata" in step:
                print(f"Metadata: {step['Metadata']}")
        elif step_status == "Succeeded":
             # If it has a job ARN, print it so user can check logs manually if they want
            metadata = step.get("Metadata", {})
            for key, value in metadata.items():
                if isinstance(value, dict) and "Arn" in value:
                    print(f"Associated Resource ARN: {value['Arn']}")

if __name__ == "__main__":
    pipeline_name = "AbalonePipeline"
    debug_last_pipeline_execution(pipeline_name)
