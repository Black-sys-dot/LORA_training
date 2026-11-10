import sagemaker
from sagemaker.pytorch import PyTorch
import boto3

boto_session = boto3.Session(region_name="us-east-1")
sagemaker_session = sagemaker.Session(boto_session=boto_session)

iam = boto_session.client('iam')
roles = iam.list_roles()['Roles']
sagemaker_roles = [r for r in roles if 'sagemaker' in r['RoleName'].lower() or 'execution' in r['RoleName'].lower()]
role = sagemaker_roles[0]['Arn'] if sagemaker_roles else iam.get_role(RoleName='SageMakerExecutionRole')['Role']['Arn']

estimator = PyTorch(
    entry_point="infer_gojo.py",
    source_dir="codes/sagemaker_src",
    role=role,
    sagemaker_session=sagemaker_session,
    framework_version="2.1",
    py_version="py310",
    instance_count=1,
    instance_type="ml.g5.xlarge", # Fast A10G GPU
    disable_profiler=True,
    debugger_hook_config=False,
)

print("Submitting Gojo LoRA inference job to AWS SageMaker...")
# Fit with no input data needed since weights are bundled in source_dir
estimator.fit(wait=False)

job_name = estimator.latest_training_job.name
print(f"AWS Inference Job Started! Name: {job_name}")
print("The EC2 instance will automatically terminate as soon as the 3 images are generated.")
