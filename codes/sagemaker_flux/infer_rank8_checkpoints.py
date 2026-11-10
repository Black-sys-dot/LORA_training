#!/usr/bin/env python3
"""Run FLUX.1-dev inference on SageMaker across 5 checkpoints (100, 200, 300, 400, 500)
for 3 benchmark prompts with fixed seeds.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Install necessary libraries inside container
print("Installing diffusers, transformers, accelerate, peft...", flush=True)
subprocess.run([
    sys.executable, "-m", "pip", "install", "--no-cache-dir",
    "diffusers>=0.30.0,<0.32.0", "transformers>=4.44.0", "accelerate>=0.33.0,<1.0.0",
    "sentencepiece", "protobuf", "peft"
], check=True)

import torch
from diffusers import FluxPipeline
import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET = os.environ.get("SAGEMAKER_BUCKET", "sagemaker-us-east-1-YOUR_ACCOUNT_ID")
S3_OUTPUT_PREFIX = "results/rank8_progression"

OUTPUT_DIR = Path("/opt/ml/model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_RESULTS_DIR = Path("/opt/ml/output_images")
LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS_DIR = Path("/opt/ml/input/data/checkpoints")

PROMPTS = [
    {
        "id": 1,
        "name": "combat_hollow_purple",
        "prompt": "A dynamic action portrait of Satoru Gojo preparing his Hollow Purple technique, intense glowing violet energy swirling between his palms, floating above a shattered city crater, cosmic distortion and debris in the air, dramatic anime masterpiece.",
        "seed": 2024,
    },
    {
        "id": 2,
        "name": "casual_tokyo_cafe",
        "prompt": "Satoru Gojo sitting relaxed at an outdoor Tokyo cafe during a gentle rain, wearing an oversized black wool sweater, holding a warm cup of coffee, raindrops on the window, soft cinematic ambient lighting.",
        "seed": 2041,
    },
    {
        "id": 3,
        "name": "style_wild_west",
        "prompt": "Satoru Gojo reimagined as a rugged Wild West gunslinger, wearing a brown leather duster coat, holster on his hip, wide-brim cowboy hat slightly tilted showing his icy-blue eyes, dusty frontier town at golden hour.",
        "seed": 2058,
    },
]

STEPS_TO_FILES = {
    100: "gojo_lora_rank8_500s_000000100.safetensors",
    200: "gojo_lora_rank8_500s_000000200.safetensors",
    300: "gojo_lora_rank8_500s_000000300.safetensors",
    400: "gojo_lora_rank8_500s_000000400.safetensors",
    500: "gojo_lora_rank8_500s.safetensors",
}

def main():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if token:
        try:
            from huggingface_hub import login
            login(token=token)
            print("Successfully authenticated to HuggingFace Hub.", flush=True)
        except Exception as e:
            print(f"HF Hub login warning: {e}", flush=True)

    s3_client = boto3.client("s3", region_name=REGION)

    print("\nAvailable files in CHECKPOINTS_DIR:", flush=True)
    for p in CHECKPOINTS_DIR.glob("*"):
        print(f"  {p.name} ({p.stat().st_size} bytes)", flush=True)

    print("\nLoading FLUX.1-dev pipeline with bfloat16...", flush=True)
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    )
    print("Enabling model CPU offload for optimal VRAM efficiency on A10G...", flush=True)
    pipe.enable_model_cpu_offload()
    print(f"Pipeline ready in {time.time() - t0:.1f}s.\n", flush=True)

    total_images = len(STEPS_TO_FILES) * len(PROMPTS)
    completed = 0

    for step, filename in sorted(STEPS_TO_FILES.items()):
        ckpt_path = CHECKPOINTS_DIR / filename
        if not ckpt_path.exists():
            # Try finding recursively
            matches = list(CHECKPOINTS_DIR.rglob(filename))
            if matches:
                ckpt_path = matches[0]
            else:
                print(f"ERROR: Checkpoint file not found: {filename}, skipping step {step}", flush=True)
                continue

        print(f"\n=======================================================", flush=True)
        print(f"Loading LoRA Checkpoint: Step {step} ({ckpt_path.name})", flush=True)
        print(f"=======================================================", flush=True)

        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

        pipe.load_lora_weights(str(ckpt_path), adapter_name="gojo_rank8")

        for p_info in PROMPTS:
            completed += 1
            p_id = p_info["id"]
            p_name = p_info["name"]
            prompt = p_info["prompt"]
            seed = p_info["seed"]

            out_filename = f"step_{step:04d}_prompt_{p_id:02d}_{p_name}.png"
            local_save_path = LOCAL_RESULTS_DIR / out_filename
            model_save_path = OUTPUT_DIR / out_filename

            print(f"\n[{completed}/{total_images}] Step {step} | Prompt {p_id}: {p_name} (seed={seed})...", flush=True)
            t_gen = time.time()
            generator = torch.Generator("cuda").manual_seed(seed)

            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=30,
                max_sequence_length=512,
                joint_attention_kwargs={"scale": 1.0},
                generator=generator,
            ).images[0]

            image.save(local_save_path)
            image.save(model_save_path)
            print(f"Generated in {time.time() - t_gen:.1f}s -> Saved: {out_filename}", flush=True)

            # Upload live to S3
            try:
                s3_key = f"{S3_OUTPUT_PREFIX}/{out_filename}"
                s3_client.upload_file(str(local_save_path), BUCKET, s3_key)
                print(f"Uploaded to S3: s3://{BUCKET}/{s3_key}", flush=True)
            except Exception as e:
                print(f"S3 upload error: {e}", flush=True)

    print("\n=======================================================", flush=True)
    print("All 15 comparison images generated and uploaded successfully!", flush=True)
    print("=======================================================", flush=True)

if __name__ == "__main__":
    main()
