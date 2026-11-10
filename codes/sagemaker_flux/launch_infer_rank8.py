#!/usr/bin/env python3
"""Launch SageMaker inference job to generate 15 images:
3 prompts across 5 checkpoints (100, 200, 300, 400, 500) for FLUX LoRA Rank 8.
Runs on ml.g5.4xlarge (1x NVIDIA A10G 24GB VRAM, 64GB host RAM).
"""

import os
import sys
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
import boto3
from dotenv import load_dotenv

REGION = os.environ.get("AWS_REGION", "us-east-1")
ACCOUNT = os.environ.get("AWS_ACCOUNT_ID", "YOUR_AWS_ACCOUNT_ID")
BUCKET = os.environ.get("SAGEMAKER_BUCKET", f"sagemaker-{REGION}-{ACCOUNT}")
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", f"arn:aws:iam::{ACCOUNT}:role/SageMakerExecutionRole")
IMAGE = f"763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker"
CODE_DIR = Path(__file__).resolve().parent

def load_hf_token() -> str:
    load_dotenv()
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        raise RuntimeError("HF_TOKEN could not be loaded from .env")
    return tok

def upload_infer_code(s3_client) -> str:
    tar_path = Path("/tmp/flux_infer_rank8.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        infer_script = CODE_DIR / "infer_rank8_checkpoints.py"
        tar.add(infer_script, arcname="infer_rank8_checkpoints.py")
    key = "code/flux_infer_rank8.tar.gz"
    s3_client.upload_file(str(tar_path), BUCKET, key)
    uri = f"s3://{BUCKET}/{key}"
    print(f"Uploaded inference source bundle to {uri}", flush=True)
    return uri

def launch_infer_job():
    token = load_hf_token()
    session = boto3.Session(region_name=REGION)
    s3 = session.client("s3")
    sm = session.client("sagemaker")

    code_uri = upload_infer_code(s3)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"gojo-flux-rank8-infer-{stamp}"
    instance_type = "ml.g5.2xlarge"

    print(f"\nSubmitting SageMaker Inference Job: {job_name} ({instance_type})...", flush=True)

    hyperparameters = {
        "sagemaker_program": json.dumps("infer_rank8_checkpoints.py"),
        "sagemaker_submit_directory": json.dumps(code_uri),
        "sagemaker_container_log_level": json.dumps(20),
        "sagemaker_region": json.dumps(REGION),
        "sagemaker_job_name": json.dumps(job_name),
    }

    response = sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=ROLE_ARN,
        AlgorithmSpecification={
            "TrainingImage": IMAGE,
            "TrainingInputMode": "File",
        },
        InputDataConfig=[
            {
                "ChannelName": "checkpoints",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{BUCKET}/checkpoints/lora_rank8_500/output/gojo_lora_rank8_500s/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "CompressionType": "None",
                "RecordWrapperType": "None",
            }
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/results/rank8_infer_job_output"},
        ResourceConfig={
            "InstanceCount": 1,
            "InstanceType": instance_type,
            "VolumeSizeInGB": 100,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600}, # 1 hour max
        HyperParameters=hyperparameters,
        Environment={
            "SAGEMAKER_PROGRAM": "infer_rank8_checkpoints.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": code_uri,
            "SAGEMAKER_REGION": REGION,
            "TORCHDYNAMO_DISABLE": "1",
            "TORCH_COMPILE_DISABLE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_TOKEN": token,
            "HUGGING_FACE_HUB_TOKEN": token,
        },
        ProfilerConfig={"DisableProfiler": True},
        EnableNetworkIsolation=False,
        Tags=[
            {"Key": "project", "Value": "gojo-flux-rank8"},
            {"Key": "task", "Value": "inference-progression"},
        ],
    )

    print(f"-> Successfully created SageMaker job: {job_name}!")
    print(f"-> ARN: {response['TrainingJobArn']}")
    print(f"-> Direct S3 Output: s3://{BUCKET}/results/rank8_progression/")
    return job_name

if __name__ == "__main__":
    launch_infer_job()
