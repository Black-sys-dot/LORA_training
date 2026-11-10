#!/usr/bin/env python3
"""Launch SageMaker Training Job for FLUX.1-dev LoRA Rank 8 (500 steps).
Runs on ml.g5.4xlarge (NVIDIA A10G 24GB VRAM, 64GB RAM) in us-east-1.
"""

import os
import sys
import json
import time
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

def upload_code(s3_client) -> str:
    tar_path = Path("/tmp/flux_sourcedir_rank8.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for p in CODE_DIR.iterdir():
            if p.name.startswith(".") or p.suffix not in {".py", ".yaml"}:
                continue
            tar.add(p, arcname=p.name)
    key = "code/flux_sourcedir_rank8.tar.gz"
    s3_client.upload_file(str(tar_path), BUCKET, key)
    uri = f"s3://{BUCKET}/{key}"
    print(f"Uploaded source code bundle to {uri}", flush=True)
    return uri

def launch_rank8_job():
    token = load_hf_token()
    session = boto3.Session(region_name=REGION)
    s3 = session.client("s3")
    sm = session.client("sagemaker")

    code_uri = upload_code(s3)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"gojo-flux-lora-rank8-{stamp}"
    run_name = "gojo_lora_rank8_500s"
    instance_type = "ml.g5.4xlarge"

    print(f"\nSubmitting SageMaker Training Job: {job_name} ({instance_type})...", flush=True)

    hyperparameters = {
        "sagemaker_program": json.dumps("train.py"),
        "sagemaker_submit_directory": json.dumps(code_uri),
        "sagemaker_container_log_level": json.dumps(20),
        "sagemaker_region": json.dumps(REGION),
        "sagemaker_job_name": json.dumps(job_name),
        "adapter_type": json.dumps("lora"),
        "run_name": json.dumps(run_name),
        "resume_step": json.dumps("0"),
        "target_steps": json.dumps("500"),
        "save_every": json.dumps("100"),
        "rank": json.dumps("8"),
        "alpha": json.dumps("8"),
        "lr": json.dumps("4e-4"),
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
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{BUCKET}/gojo/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "CompressionType": "None",
                "RecordWrapperType": "None",
            }
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/output/lora_rank8"},
        ResourceConfig={
            "InstanceCount": 1,
            "InstanceType": instance_type,
            "VolumeSizeInGB": 100,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 10800}, # 3 hours max
        CheckpointConfig={
            "S3Uri": f"s3://{BUCKET}/checkpoints/lora_rank8_500",
            "LocalPath": "/opt/ml/checkpoints",
        },
        HyperParameters=hyperparameters,
        Environment={
            "SAGEMAKER_PROGRAM": "train.py",
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
            {"Key": "adapter", "Value": "lora"},
            {"Key": "rank", "Value": "8"},
        ],
    )

    print(f"-> Successfully created SageMaker job: {job_name}!")
    print(f"-> ARN: {response['TrainingJobArn']}")
    print(f"-> Checkpoint Destination: s3://{BUCKET}/checkpoints/lora_rank8_500/")
    return job_name

if __name__ == "__main__":
    launch_rank8_job()
