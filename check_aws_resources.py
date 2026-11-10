import boto3

def check_resources():
    ec2_client = boto3.client('ec2', region_name='us-east-1')
    regions = [region['RegionName'] for region in ec2_client.describe_regions()['Regions']]
    
    found_ec2 = []
    found_sagemaker = []
    
    for region in regions:
        print(f"Checking {region}...")
        try:
            # Check EC2
            ec2 = boto3.client('ec2', region_name=region)
            instances = ec2.describe_instances(Filters=[{'Name': 'instance-state-name', 'Values': ['running', 'pending']}])
            for res in instances.get('Reservations', []):
                for inst in res.get('Instances', []):
                    found_ec2.append((region, inst['InstanceId'], inst['InstanceType']))
            
            # Check SageMaker Training Jobs
            sm = boto3.client('sagemaker', region_name=region)
            jobs = sm.list_training_jobs(StatusEquals='InProgress')
            for job in jobs.get('TrainingJobSummaries', []):
                found_sagemaker.append((region, job['TrainingJobName']))
        except Exception as e:
            print(f"Error checking {region}: {e}")

    print("\n--- Summary ---")
    if not found_ec2:
        print("No running EC2 instances found across all regions.")
    else:
        for r, i, t in found_ec2:
            print(f"Running EC2 instance in {r}: {i} ({t})")
            
    if not found_sagemaker:
        print("No running SageMaker jobs found across all regions.")
    else:
        for r, j in found_sagemaker:
            print(f"Running SageMaker job in {r}: {j}")

if __name__ == "__main__":
    check_resources()
