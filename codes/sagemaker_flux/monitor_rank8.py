#!/usr/bin/env python3
"""Monitor the active FLUX LoRA Rank 8 training job on SageMaker.
"""

import boto3
from datetime import datetime

REGION = "us-east-1"
JOB_NAME = "gojo-flux-lora-rank8-20261014-075336"

def check_status():
    session = boto3.Session(region_name=REGION)
    sm = session.client("sagemaker")
    logs = session.client("logs")
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking Job Status for {JOB_NAME}:\n" + "-"*65)
    try:
        desc = sm.describe_training_job(TrainingJobName=JOB_NAME)
    except Exception as e:
        print(f"Error fetching status: {e}")
        return
        
    status = desc["TrainingJobStatus"]
    secondary = desc.get("SecondaryStatus", "Unknown")
    trans = desc.get("SecondaryStatusTransitions", [])
    msg = trans[-1].get("StatusMessage", "") if trans else ""
    
    print(f"Status: {status} ({secondary})")
    if msg:
        print(f"Detail: {msg}")
        
    if status == "InProgress":
        try:
            log_group = "/aws/sagemaker/TrainingJobs"
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
                    limit=5,
                    startFromHead=False
                ).get("events", [])
                
                print("\nLatest CloudWatch logs:")
                for ev in events:
                    msg_line = ev["message"].strip()
                    if msg_line:
                        print(f"  {msg_line}")
        except Exception as e:
            print(f"Logs: {e}")

if __name__ == "__main__":
    check_status()
