#!/usr/bin/env python3
"""Autonomous Background Monitor & Downloader for Rank 8 Checkpoints Inference.
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
JOB_NAME = "gojo-flux-rank8-infer-20261014-143311"
BUCKET = os.environ.get("SAGEMAKER_BUCKET", "sagemaker-us-east-1-YOUR_ACCOUNT_ID")
S3_PREFIX = "results/rank8_progression/"
LOCAL_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "rank8_progression"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "infer_monitor.log"

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def sync_images_from_s3(s3_client):
    try:
        resp = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=S3_PREFIX)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            filename = Path(key).name
            if filename.endswith(".png"):
                dest = LOCAL_DIR / filename
                if not dest.exists():
                    s3_client.download_file(BUCKET, key, str(dest))
                    log(f"📥 Downloaded to local project: {filename}")
    except Exception as e:
        log(f"Sync error: {e}")

def run_monitor():
    log(f"Starting Inference Monitor for {JOB_NAME}...")
    session = boto3.Session(region_name=REGION)
    sm = session.client("sagemaker")
    logs = session.client("logs")
    s3 = session.client("s3")
    log_group = "/aws/sagemaker/TrainingJobs"

    last_log_seen = ""

    while True:
        try:
            desc = sm.describe_training_job(TrainingJobName=JOB_NAME)
            status = desc["TrainingJobStatus"]
            secondary = desc.get("SecondaryStatus", "Unknown")
            trans = desc.get("SecondaryStatusTransitions", [])
            msg = trans[-1].get("StatusMessage", "") if trans else ""

            # Check CloudWatch logs
            latest_cw_msg = ""
            try:
                streams = logs.describe_log_streams(
                    logGroupName=log_group,
                    logStreamNamePrefix=JOB_NAME,
                    limit=5
                ).get("logStreams", [])

                if streams:
                    streams.sort(key=lambda x: x.get("lastEventTimestamp", 0), reverse=True)
                    stream_name = streams[0]["logStreamName"]
                    events = logs.get_log_events(
                        logGroupName=log_group,
                        logStreamName=stream_name,
                        limit=8,
                        startFromHead=False
                    ).get("events", [])
                    
                    for ev in reversed(events):
                        line = ev["message"].strip()
                        if line and line != "#033[A":
                            latest_cw_msg = line
                            break
            except Exception:
                pass

            if latest_cw_msg and latest_cw_msg != last_log_seen:
                last_log_seen = latest_cw_msg
                log(f"[{secondary}] {msg} | Log: {latest_cw_msg}")
            else:
                log(f"[{secondary}] {msg}")

            # Check for newly generated images on S3
            sync_images_from_s3(s3)

            if status == "Completed":
                log("🎉 Inference Job COMPLETED successfully! Syncing final images...")
                sync_images_from_s3(s3)
                break
            elif status in ("Failed", "Stopped"):
                failure_reason = desc.get("FailureReason", "No failure reason provided")
                log(f"⚠️ Inference Job ended with status: {status}. Reason: {failure_reason}")
                break

        except Exception as e:
            log(f"Error during poll: {e}")

        time.sleep(30)

if __name__ == "__main__":
    run_monitor()
