import sagemaker
from sagemaker.pytorch import PyTorch
import boto3

# Initialize session
boto_session = boto3.Session(region_name="us-east-1")
sagemaker_session = sagemaker.Session(boto_session=boto_session)

# The IAM role needs to have SageMaker execution permissions. 
# We can try to get the execution role, but if we are running outside SageMaker we might have to pass an ARN.
try:
    role = sagemaker.get_execution_role()
except ValueError:
    # If running locally, assume the AWS CLI user has a role or is admin, but SageMaker requires an IAM Role ARN.
    # We will try to find a SageMaker execution role in the account.
    iam = boto_session.client('iam')
    roles = iam.list_roles()['Roles']
    sagemaker_roles = [r for r in roles if 'sagemaker' in r['RoleName'].lower() or 'execution' in r['RoleName'].lower()]
    if sagemaker_roles:
        role = sagemaker_roles[0]['Arn']
    else:
        role = iam.get_role(RoleName='SageMakerExecutionRole')['Role']['Arn'] # hope it exists

print(f"Using role: {role}")

bucket = os.environ.get("SAGEMAKER_BUCKET", "sagemaker-us-east-1-YOUR_ACCOUNT_ID")
s3_input_data = f"s3://{bucket}/gojo/"

# We configure the PyTorch estimator
estimator = PyTorch(
    entry_point="train.py",
    source_dir="codes/sagemaker_src",
    role=role,
    sagemaker_session=sagemaker_session,
    framework_version="2.1",
    py_version="py310",
    instance_count=1,
    instance_type="ml.g5.xlarge", # A10G Ampere GPU (24GB VRAM, fast native FlashAttention)
    hyperparameters={
        "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
        "train_data_dir": "/opt/ml/input/data/training", # SageMaker downloads the S3 data here
        "dataloader_num_workers": 2,
        "resolution": 512,
        "train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "max_train_steps": 1000,
        "learning_rate": 1e-4,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "output_dir": "/opt/ml/model", # SageMaker uploads this directory back to S3 automatically!
        "checkpointing_steps": 1000,
        "mixed_precision": "fp16",
        "seed": 42
    },
    disable_profiler=True,
    debugger_hook_config=False,
)

print("Submitting training job to SageMaker...")
# Fit the estimator, which launches the EC2 instance, downloads data, runs training, uploads model, and kills instance
estimator.fit({"training": s3_input_data}, wait=False)

print(f"Job started! Name: {estimator.latest_training_job.name}")
print("The EC2 instance will automatically terminate when it finishes or fails. No credits will be leaked!")
