#!/usr/bin/env python3
"""Autonomous Monitor & Auto-Restart Daemon for FLUX Adapter Training Jobs on SageMaker.

Monitors:
  - DoRA (ml.g5.2xlarge)
  - LoKr (ml.g5.4xlarge)
  - LoRA (ml.g5.xlarge)

Features:
  - Tracks training progress (step, loss, speed) via CloudWatch logs.
  - Monitors S3 checkpoint syncs (every 100 steps).
  - Detects container crashes or SageMaker job failures.
  - On failure: parses the latest saved S3 checkpoint, re-seeds if needed,
    and automatically relaunches the job to seamlessly continue to 1500 steps.
  - Maintains persistent state in jobs_status.json and monitor.log.
"""

import os
import sys
import time
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import boto3
from dotenv import load_dotenv

REGION = os.environ.get("AWS_REGION", "us-east-1")
ACCOUNT = os.environ.get("AWS_ACCOUNT_ID", "YOUR_AWS_ACCOUNT_ID")
BUCKET = os.environ.get("SAGEMAKER_BUCKET", f"sagemaker-{REGION}-{ACCOUNT}")
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", f"arn:aws:iam::{ACCOUNT}:role/SageMakerExecutionRole")
IMAGE = f"763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker"

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "jobs_status.json"
LOG_FILE = BASE_DIR / "monitor.log"

INITIAL_JOBS = {
    "dora": {
        "tag": "DoRA",
        "job_name": "gojo-flux-dora-20261011-160821",
        "instance_type": "ml.g5.2xlarge",
        "adapter_type": "dora",
        "run_name": "gojo_dora_rank4_1500s",
        "target_steps": 1500,
        "initial_resume_step": 600,
        "current_step": 600,
        "status": "InProgress",
        "restarts": 0,
    },
    "lokr": {
        "tag": "LoKr",
        "job_name": "gojo-flux-lokr-20261011-160821",
        "instance_type": "ml.g5.4xlarge",
        "adapter_type": "lokr",
        "run_name": "gojo_lycoris_rank4_1500s",
        "target_steps": 1500,
        "initial_resume_step": 600,
        "current_step": 600,
        "status": "InProgress",
        "restarts": 0,
    },
    "lora": {
        "tag": "LoRA",
        "job_name": "gojo-flux-lora-20261011-160821",
        "instance_type": "ml.g5.12xlarge",
        "adapter_type": "lora",
        "run_name": "gojo_lora_rank4_1500s",
        "target_steps": 1500,
        "initial_resume_step": 1000,
        "current_step": 1000,
        "status": "InProgress",
        "restarts": 0,
    },
}

def log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{now}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def load_hf_token() -> str:
    load_dotenv()
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        raise RuntimeError("HF_TOKEN could not be loaded")
    return tok

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return INITIAL_JOBS

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def get_code_uri(s3_client) -> str:
    # Ensure source code bundle is up to date
    return f"s3://{BUCKET}/code/flux_sourcedir.tar.gz"

def get_latest_s3_checkpoint(s3_client, adapter_type: str, run_name: str) -> tuple[int, list[str]]:
    """Check checkpoints directory for the latest step and list of safetensors files."""
    prefix = f"checkpoints/{adapter_type}_1500/"
    res = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    contents = res.get("Contents", [])
    
    files = [obj["Key"] for obj in contents]
    highest_step = 0
    step_pattern = re.compile(r"_(\d{9})\.safetensors$")
    
    for key in files:
        m = step_pattern.search(key)
        if m:
            step = int(m.group(1))
            if step > highest_step:
                highest_step = step
                
    return highest_step, files

def restart_job(sm_client, s3_client, job_info: dict) -> str:
    """Relaunch a failed job from the latest checkpoint on S3."""
    adapter_type = job_info["adapter_type"]
    run_name = job_info["run_name"]
    inst_type = job_info["instance_type"]
    tag = job_info["tag"]
    
    # 1. Inspect latest checkpoints
    latest_step, files = get_latest_s3_checkpoint(s3_client, adapter_type, run_name)
    resume_step = max(latest_step, job_info.get("initial_resume_step", 600))
    
    log(f"RESTARTING {tag} ({adapter_type}): Resuming from step {resume_step}...")
    
    # If S3 checkpoint has an optimizer and safetensors, update seed prefix
    if latest_step > job_info.get("initial_resume_step", 600):
        log(f"Syncing latest step {latest_step} files to seed folder s3://{BUCKET}/gojo_seeds/{adapter_type}/...")
        # Copy latest weights and optimizer to seed
        prefix = f"checkpoints/{adapter_type}_1500/output/{run_name}/"
        src_res = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        for obj in src_res.get("Contents", []):
            k = obj["Key"]
            fname = k.split("/")[-1]
            if fname.endswith(f"_{latest_step:09d}.safetensors") or fname == "optimizer.pt" or fname.endswith(".safetensors"):
                s3_client.copy_object(
                    Bucket=BUCKET,
                    CopySource={"Bucket": BUCKET, "Key": k},
                    Key=f"gojo_seeds/{adapter_type}/{fname}"
                )
                log(f"  Copied {fname} to seed directory.")
                
    # 2. Formulate new job name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    new_job_name = f"gojo-flux-{adapter_type}-r{job_info['restarts']+1}-{stamp}"
    code_uri = get_code_uri(s3_client)
    token = load_hf_token()
    
    hyperparameters = {
        "sagemaker_program": json.dumps("train.py"),
        "sagemaker_submit_directory": json.dumps(code_uri),
        "sagemaker_container_log_level": json.dumps(20),
        "sagemaker_region": json.dumps(REGION),
        "sagemaker_job_name": json.dumps(new_job_name),
        "adapter_type": json.dumps(adapter_type),
        "run_name": json.dumps(run_name),
        "resume_step": json.dumps(str(resume_step)),
        "target_steps": json.dumps(str(job_info["target_steps"])),
        "save_every": json.dumps("100"),
        "rank": json.dumps("4"),
        "lr": json.dumps("4e-4"),
    }
    
    response = sm_client.create_training_job(
        TrainingJobName=new_job_name,
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
            "InstanceType": inst_type,
            "VolumeSizeInGB": 100,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 18000},
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
            {"Key": "restarted", "Value": "true"},
        ],
    )
    
    log(f"SUCCESS: Launched replacement job {new_job_name} ({inst_type})! ARN: {response['TrainingJobArn']}")
    return new_job_name

def monitor_loop(continuous: bool = True, interval_sec: int = 40):
    session = boto3.Session(region_name=REGION)
    sm = session.client("sagemaker")
    logs = session.client("logs")
    s3 = session.client("s3")
    
    state = load_state()
    log("Starting Autonomous Training Monitor & Crash Guardian...")
    
    while True:
        all_completed = True
        
        for key, job in list(state.items()):
            tag = job["tag"]
            job_name = job["job_name"]
            status = job.get("status", "Unknown")
            
            if status == "Completed":
                continue
                
            all_completed = False
            
            try:
                desc = sm.describe_training_job(TrainingJobName=job_name)
            except Exception as e:
                log(f"[{tag}] Error describing job {job_name}: {e}")
                continue
                
            cur_status = desc["TrainingJobStatus"]
            secondary = desc.get("SecondaryStatus", "Unknown")
            trans = desc.get("SecondaryStatusTransitions", [])
            last_msg = trans[-1].get("StatusMessage", "") if trans else ""
            job["status"] = cur_status
            
            # Fetch latest log line
            last_log = ""
            try:
                log_group = "/aws/sagemaker/TrainingJobs"
                streams = logs.describe_log_streams(
                    logGroupName=log_group,
                    logStreamNamePrefix=job_name,
                    limit=5
                ).get("logStreams", [])
                
                if streams:
                    streams.sort(key=lambda x: x.get("lastEventTimestamp", 0), reverse=True)
                    s_name = streams[0]["logStreamName"]
                    events = logs.get_log_events(
                        logGroupName=log_group,
                        logStreamName=s_name,
                        limit=5,
                        startFromHead=False
                    ).get("events", [])
                    for ev in reversed(events):
                        msg = ev["message"].strip()
                        if msg:
                            last_log = msg
                            # Try to extract current step if reported by ai-toolkit
                            step_match = re.search(r"step\s+(\d+)|(\d+)/(\d+)\s+\[", msg, re.IGNORECASE)
                            if step_match:
                                val = step_match.group(1) or step_match.group(2)
                                job["current_step"] = int(val)
                            break
            except Exception:
                pass
                
            # Check S3 checkpoints
            try:
                s3_step, ckpt_files = get_latest_s3_checkpoint(s3, job["adapter_type"], job["run_name"])
                if s3_step > 0:
                    job["latest_s3_checkpoint_step"] = s3_step
            except Exception:
                pass

            if cur_status == "InProgress":
                log(f"[{tag}] {job_name} | {secondary}: {last_msg[:60]} | Log: {last_log[:80]}")
                
            elif cur_status == "Completed":
                log(f"[{tag}] {job_name} COMPLETED! Artifacts synced to S3.")
                job["status"] = "Completed"
                job["completed_at"] = datetime.now(timezone.utc).isoformat()
                
            elif cur_status == "Failed":
                failure_reason = desc.get("FailureReason", "Unknown failure")
                log(f"[{tag}] *** ALERT: Job {job_name} FAILED! ***")
                log(f"[{tag}] Failure Reason: {failure_reason}")
                log(f"[{tag}] Last Log: {last_log}")
                
                # Check restart limits
                if job.get("restarts", 0) < 3:
                    job["restarts"] = job.get("restarts", 0) + 1
                    try:
                        new_job_name = restart_job(sm, s3, job)
                        job["job_name"] = new_job_name
                        job["status"] = "InProgress"
                        job["last_restart_at"] = datetime.now(timezone.utc).isoformat()
                        log(f"[{tag}] Successfully restarted as {new_job_name} (Attempt {job['restarts']}/3)")
                    except Exception as err:
                        log(f"[{tag}] CRITICAL: Failed to auto-restart job: {err}")
                else:
                    log(f"[{tag}] Maximum restart limit (3) reached. Manual intervention required.")
                    
            save_state(state)
            
        if all_completed:
            log("All 3 training jobs have COMPLETED successfully!")
            break
            
        if not continuous:
            break
            
        time.sleep(interval_sec)

if __name__ == "__main__":
    is_continuous = "--once" not in sys.argv
    monitor_loop(continuous=is_continuous)
