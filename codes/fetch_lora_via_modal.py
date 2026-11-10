import modal
import os
import boto3
import io
import tarfile

app = modal.App("extract-lora-weights")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("boto3")
)

session = boto3.Session()
creds = session.get_credentials().get_frozen_credentials()

aws_secret = modal.Secret.from_dict({
    "AWS_ACCESS_KEY_ID": creds.access_key,
    "AWS_SECRET_ACCESS_KEY": creds.secret_key,
    "AWS_DEFAULT_REGION": "us-east-1"
})

@app.function(
    image=image,
    secrets=[aws_secret],
    timeout=180
)
def extract_lora_from_s3(bucket: str, key: str) -> dict[str, bytes]:
    import boto3
    import tarfile
    import io

    print(f"Downloading s3://{bucket}/{key} inside cloud datacenter...")
    s3 = boto3.client("s3")
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    buf.seek(0)
    print("Download complete! Opening tar archive in-memory...")

    extracted = {}
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name == "pytorch_lora_weights.safetensors" or member.name == "./pytorch_lora_weights.safetensors":
                print(f"Extracting target weights: {member.name} ({member.size} bytes)...")
                f = tar.extractfile(member)
                if f:
                    extracted["pytorch_lora_weights.safetensors"] = f.read()
                break

    print(f"Successfully extracted {list(extracted.keys())}!")
    return extracted

@app.local_entrypoint()
def main():
    bucket = os.environ.get("SAGEMAKER_BUCKET", "sagemaker-us-east-1-YOUR_ACCOUNT_ID")
    key = "pytorch-training-2026-10-11-04-32-11-110/output/model.tar.gz"
    output_dir = "models/sd15_lora"
    os.makedirs(output_dir, exist_ok=True)

    print("Requesting fast extraction via Modal cloud...")
    extracted_files = extract_lora_from_s3.remote(bucket, key)

    for filename, data in extracted_files.items():
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved: {out_path} ({len(data)} bytes)")

    print("Done! Adapter weights ready.")
