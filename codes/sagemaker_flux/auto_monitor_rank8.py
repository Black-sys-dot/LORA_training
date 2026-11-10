#!/usr/bin/env python3
"""Autonomous Background Monitor for FLUX LoRA Rank 8 training on SageMaker.
"""

import sys
import time
import json
import re
from datetime import datetime
from pathlib import Path
import boto3

REGION = "us-east-1"
JOB_NAME = "gojo-flux-lora-rank8-20261014-075336"
BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "rank8_monitor.log"

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def run_monitor():
    log(f"Starting Rank 8 Autonomous Monitor for {JOB_NAME}...")
    session = boto3.Session(region_name=REGION)
    sm = session.client("sagemaker")
    logs = session.client("logs")
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

            if status == "Completed":
                log("🎉 Training Job COMPLETED successfully! Checkpoints are saved on S3.")
                break
            elif status in ("Failed", "Stopped"):
                failure_reason = desc.get("FailureReason", "No failure reason provided")
                log(f"⚠️ Training Job ended with status: {status}. Reason: {failure_reason}")
                break

        except Exception as e:
            log(f"Error during poll: {e}")

        time.sleep(30)

if __name__ == "__main__":
    run_monitor()
