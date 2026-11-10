import modal
import os
import subprocess

app = modal.App("flux-lora-training-gojo-rank4")

# We build a Modal image that installs AI Toolkit and its dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone https://github.com/ostris/ai-toolkit.git /ai-toolkit",
        "cd /ai-toolkit && git submodule update --init --recursive",
        "cd /ai-toolkit && pip install -r requirements.txt",
        "pip install diffusers[torch] peft accelerate transformers huggingface_hub torchaudio torchvision"
    )
)

# Volume to store the dataset and output checkpoints
volume = modal.Volume.from_name("flux-lora-volume", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-80GB", # Using A100-80GB to bypass H100 Hopper bug
    timeout=7200, # 2 hours max (container automatically shuts down when function ends)
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")] # Needs HF_TOKEN to download FLUX [dev]
)
def train_model():
    import yaml
    
    # AI Toolkit configuration YAML
    config = {
        "job": "extension",
        "config": {
            "name": "gojo_lora_rank4",
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": "/data/output",
                    "device": "cuda:0",
                    "network": {
                        "type": "lora",
                        "linear": 4,
                        "linear_alpha": 4
                    },
                    "save": {
                        "dtype": "bfloat16",
                        "save_every": 250,
                        "max_step_saves_to_keep": 4
                    },
                    "datasets": [
                        {
                            "folder_path": "/data/dataset/gojo",
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "shuffle_tokens": False,
                            "cache_latents_to_disk": True,
                            "resolution": [512, 768, 1024]
                        }
                    ],
                    "train": {
                        "batch_size": 1,
                        "steps": 1000,
                        "gradient_accumulation_steps": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "content_or_style": "balanced",
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw",
                        "lr": 4e-4,
                        "ema_config": {
                            "use_ema": True,
                            "ema_decay": 0.99
                        },
                        "dtype": "bf16"
                    },
                    "model": {
                        "name_or_path": "black-forest-labs/FLUX.1-dev",
                        "is_flux": True,
                        "quantize": False
                    }
                }
            ]
        }
    }
    
    with open("/ai-toolkit/config.yaml", "w") as f:
        yaml.dump(config, f)
        
    cmd = ["python", "run.py", "config.yaml"]
    
    # Run AI Toolkit and print output for the local dashboard to parse
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="/ai-toolkit")
    
    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)
        
    process.wait()

@app.local_entrypoint()
def main():
    print("Starting Modal training job... Remote logs will follow:")
    train_model.remote()
