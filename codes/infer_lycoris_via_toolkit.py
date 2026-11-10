import modal
import os
import subprocess

app = modal.App("flux-lycoris-sample")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone https://github.com/ostris/ai-toolkit.git /ai-toolkit",
        "cd /ai-toolkit && git submodule update --init --recursive",
        "cd /ai-toolkit && pip install -r requirements.txt",
        "pip install diffusers[torch] peft accelerate transformers huggingface_hub torchaudio torchvision lycoris_lora"
    )
)

volume = modal.Volume.from_name("flux-lora-volume")

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1200, 
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def generate_model():
    import yaml
    
    config = {
        "job": "extension",
        "config": {
            "name": "gojo_lycoris_sample",
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": "/data/output",
                    "device": "cuda:0",
                    "network": {
                        "type": "lokr",
                        "linear": 4,
                        "linear_alpha": 4,
                        # Load the fully trained weights we just created
                        "network_dir": "/data/output/gojo_lycoris_rank4" # Or we can just rely on the path if ai-toolkit supports it. Wait, the parameter in ai-toolkit is "resume"
                    },
                    "save": {
                        "dtype": "bfloat16",
                        "save_every": 1000,
                        "max_step_saves_to_keep": 1
                    },
                    "datasets": [
                        {
                            "folder_path": "/data/dataset/gojo",
                            "caption_ext": "txt",
                            "resolution": [512]
                        }
                    ],
                    "train": {
                        "batch_size": 1,
                        "steps": 1, # Just 1 step
                        "gradient_accumulation_steps": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "optimizer": "adamw",
                        "lr": 0.0, # Do not change weights
                        "dtype": "bf16"
                    },
                    "model": {
                        "name_or_path": "black-forest-labs/FLUX.1-dev",
                        "is_flux": True,
                        "quantize": False
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": 1,
                        "width": 1024,
                        "height": 1024,
                        "prompts": [
                            "Satoru Gojo drinking tea in a garden.",
                            "Satoru Gojo, adult male, pure-white hair, icy-blue eyes, wearing a casual hoodie, standing in a rainy city street.",
                            "Satoru Gojo, adult male, pure-white hair, icy-blue eyes, glowing purple energy around his hands, levitating above a crumbling temple, dramatic cinematic lighting, masterpiece illustration, 8k resolution, volumetric fog.",
                            "Satoru Gojo wearing heavy medieval knight armor holding a broadsword."
                        ],
                        "neg": "",
                        "seed": 100,
                        "walk_seed": True,
                        "guidance_scale": 3.5,
                        "sample_steps": 30
                    }
                }
            ]
        }
    }
    
    # We must patch the network resume manually or rely on ai-toolkit finding it.
    # To be perfectly safe, since ai-toolkit supports resuming via "resume": true
    # wait, "resume" usually looks for the last checkpoint in training_folder.
    # So if we set training_folder to /data/output/gojo_lycoris_rank4, it will automatically load it!
    config["config"]["process"][0]["training_folder"] = "/data/output/gojo_lycoris_rank4"
    
    with open("/ai-toolkit/config.yaml", "w") as f:
        yaml.dump(config, f)
        
    cmd = ["python", "run.py", "config.yaml"]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="/ai-toolkit")
    
    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)
        
    process.wait()
    
    # Copy the samples to a common dir
    import glob
    import shutil
    samples = glob.glob("/data/output/gojo_lycoris_rank4/gojo_lycoris_sample/samples/*.png")
    os.makedirs("/data/output/lycoris_final_samples", exist_ok=True)
    for s in samples:
        shutil.copy(s, "/data/output/lycoris_final_samples/")
        
    return True

@app.local_entrypoint()
def main():
    print("Starting Modal LyCORIS (LoKr) sampling job via toolkit bypass...")
    generate_model.remote()
