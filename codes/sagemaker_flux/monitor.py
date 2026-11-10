#!/usr/bin/env python3
"""Monitor all 3 active SageMaker training jobs.
Queries SageMaker API and CloudWatch Logs to report status, current step, and detect crashes.
"""

import boto3
from datetime import datetime

REGION = "us-east-1"
JOBS = [
    ("DoRA", "gojo-flux-dora-20261011-160821"),
    ("LoKr", "gojo-flux-lokr-20261011-160821"),
    ("LoRA", "gojo-flux-lora-20261011-160821"),
]

def check_jobs():
    session = boto3.Session(region_name=REGION)
    sm = session.client("sagemaker")
    logs = session.client("logs")
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking SageMaker Jobs Status:\n" + "-"*65)
    
    all_done = True
    any_failed = False
    
    for tag, name in JOBS:
        try:
            desc = sm.describe_training_job(TrainingJobName=name)
        except Exception as e:
            print(f"[{tag}] {name}: Error fetching status - {e}")
            continue
            
        status = desc["TrainingJobStatus"]
        secondary = desc.get("SecondaryStatus", "Unknown")
        trans = desc.get("SecondaryStatusTransitions", [])
        msg = trans[-1].get("StatusMessage", "") if trans else ""
        
        if status == "InProgress":
            all_done = False
            print(f"[{tag:4s}] {name}\n       STATUS: {status} ({secondary}) | {msg}")
            
            # Fetch last few log lines from CloudWatch
            try:
                log_group = "/aws/sagemaker/TrainingJobs"
                streams = logs.describe_log_streams(
                    logGroupName=log_group,
                    logStreamNamePrefix=name,
                    limit=5
                ).get("logStreams", [])
                
                if streams:
                    streams.sort(key=lambda x: x.get("lastEventTimestamp", 0), reverse=True)
                    stream_name = streams[0]["logStreamName"]
                    events = logs.get_log_events(
                        logGroupName=log_group,
                        logStreamName=stream_name,
                        limit=3,
                        startFromHead=False
                    ).get("events", [])
                    
                    if events:
                        last_lines = [e["message"].strip() for e in events if e["message"].strip()]
                        print(f"       LOGS: {last_lines[-1][:120] if last_lines else 'Initializing...'}")
            except Exception:
                pass
                
        elif status == "Failed":
            any_failed = True
            reason = desc.get("FailureReason", "No failure reason provided")
            print(f"[{tag:4s}] {name}\n       STATUS: *** FAILED ***\n       REASON: {reason}")
            
        elif status == "Completed":
            print(f"[{tag:4s}] {name}\n       STATUS: *** COMPLETED *** (Artifacts uploaded to S3)")
            
        print("-" * 65)
        
    return all_done, any_failed

if __name__ == "__main__":
    check_jobs()
