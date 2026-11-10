#!/usr/bin/env python3
"""Launch 3 Concurrent SageMaker Training Jobs on AWS:
1. DoRA (ml.g5.2xlarge): Resume from 600 -> 1500 steps
2. LoKr / LyCORIS (ml.g5.4xlarge): Resume from 600 -> 1500 steps
3. LoRA (ml.g5.xlarge): Resume from 1000 -> 1500 steps
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
    # Try dotenv in current dir or parents
    load_dotenv()
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        raise RuntimeError("HF_TOKEN could not be loaded from .env or environment")
    return tok

def upload_code(s3_client) -> str:
    tar_path = Path("/tmp/flux_sourcedir.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for p in CODE_DIR.iterdir():
            if p.name.startswith(".") or p.suffix not in {".py", ".yaml"}:
                continue
            tar.add(p, arcname=p.name)
    key = "code/flux_sourcedir.tar.gz"
    for attempt in range(4):
        try:
            s3_client.upload_file(str(tar_path), BUCKET, key)
            break
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2)
    uri = f"s3://{BUCKET}/{key}"
    print(f"Uploaded source code bundle to {uri}", flush=True)
    return uri

def launch_training_job(sm_client, job_name, instance_type, code_uri, token, adapter_type, run_name, resume_step, target_steps):
    print(f"\nSubmitting SageMaker Job: {job_name} ({instance_type})...", flush=True)
    
    hyperparameters = {
        "sagemaker_program": json.dumps("train.py"),
        "sagemaker_submit_directory": json.dumps(code_uri),
        "sagemaker_container_log_level": json.dumps(20),
        "sagemaker_region": json.dumps(REGION),
        "sagemaker_job_name": json.dumps(job_name),
        "adapter_type": json.dumps(adapter_type),
        "run_name": json.dumps(run_name),
        "resume_step": json.dumps(str(resume_step)),
        "target_steps": json.dumps(str(target_steps)),
        "save_every": json.dumps("100"),
        "rank": json.dumps("4"),
        "lr": json.dumps("4e-4"),
    }

    response = sm_client.create_training_job(
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
            },
            {
                "ChannelName": "seed",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{BUCKET}/gojo_seeds/{adapter_type}/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "CompressionType": "None",
                "RecordWrapperType": "None",
            }
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/output/{adapter_type}"},
        ResourceConfig={
            "InstanceCount": 1,
            "InstanceType": instance_type,
            "VolumeSizeInGB": 100,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 18000}, # 5 hours cap
        CheckpointConfig={
            "S3Uri": f"s3://{BUCKET}/checkpoints/{adapter_type}_1500",
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
            {"Key": "project", "Value": "gojo-flux-comparison"},
            {"Key": "adapter", "Value": adapter_type},
        ],
    )
    print(f"-> Successfully created {job_name}! ARN: {response['TrainingJobArn']}", flush=True)
    return response

def main():
    token = load_hf_token()
    session = boto3.Session(region_name=REGION)
    s3 = session.client("s3")
    sm = session.client("sagemaker")
    
    code_uri = upload_code(s3)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    
    launched = []
    
    jobs_config = [
        # 1. DoRA (Resume from 600 -> 1500 steps) on ml.g5.2xlarge
        {
            "job_name": f"gojo-flux-dora-{stamp}",
            "instance_type": "ml.g5.2xlarge",
            "adapter_type": "dora",
            "run_name": "gojo_dora_rank4_1500s",
            "resume_step": 600,
            "target_steps": 1500,
        },
        # 2. LyCORIS / LoKr (Resume from 600 -> 1500 steps) on ml.g5.4xlarge
        {
            "job_name": f"gojo-flux-lokr-{stamp}",
            "instance_type": "ml.g5.4xlarge",
            "adapter_type": "lokr",
            "run_name": "gojo_lycoris_rank4_1500s",
            "resume_step": 600,
            "target_steps": 1500,
        },
        # 3. LoRA (Resume from 1000 -> 1500 steps) on ml.g5.12xlarge
        {
            "job_name": f"gojo-flux-lora-{stamp}",
            "instance_type": "ml.g5.12xlarge",
            "adapter_type": "lora",
            "run_name": "gojo_lora_rank4_1500s",
            "resume_step": 1000,
            "target_steps": 1500,
        },
    ]
    
    for cfg in jobs_config:
        res = launch_training_job(
            sm,
            job_name=cfg["job_name"],
            instance_type=cfg["instance_type"],
            code_uri=code_uri,
            token=token,
            adapter_type=cfg["adapter_type"],
            run_name=cfg["run_name"],
            resume_step=cfg["resume_step"],
            target_steps=cfg["target_steps"],
        )
        launched.append((cfg["job_name"], cfg["instance_type"], cfg["adapter_type"]))
        time.sleep(2) # Stagger slightly to avoid any API throttling
        
    print("\n" + "="*60)
    print("ALL 3 SAGEMAKER JOBS LAUNCHED CONCURRENTLY ON AWS:")
    for j_name, inst, adapt in launched:
        print(f"  * {adapt.upper()}: {j_name} running on {inst}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
