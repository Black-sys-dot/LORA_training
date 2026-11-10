#!/usr/bin/env python3
"""SageMaker Entrypoint for FLUX.1-dev adapter training via ai-toolkit.
Supports resuming LoRA, DoRA, and LoKr (LyCORIS) up to step 1500.
"""

from __future__ import annotations
import os
import sys
import time
import shutil
import subprocess
import json
from pathlib import Path

TRAINING = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
SEED_DIR = Path(os.environ.get("SM_CHANNEL_SEED", "/opt/ml/input/data/seed"))
MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
CKPT_DIR = Path(os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
THIS_DIR = Path(__file__).resolve().parent
TOOLKIT_DIR = Path("/opt/ml/code/ai-toolkit")
OUTPUT_DIR = CKPT_DIR / "output"
HF_CACHE = Path("/opt/ml/hf")

# Read hyperparameters
hps_path = Path("/opt/ml/input/config/hyperparameters.json")
if hps_path.exists():
    try:
        HPS = json.loads(hps_path.read_text(encoding="utf-8"))
    except Exception:
        HPS = {}
else:
    HPS = {}

def get_hp(key: str, default: str) -> str:
    raw = HPS.get(key, os.environ.get(key.upper(), default))
    if isinstance(raw, str) and raw.startswith('"') and raw.endswith('"'):
        return json.loads(raw)
    return str(raw)

ADAPTER_TYPE = get_hp("adapter_type", "dora").lower()
RUN_NAME = get_hp("run_name", f"gojo_{ADAPTER_TYPE}_rank4_1500s")
RESUME_STEP = int(get_hp("resume_step", "600"))
TARGET_STEPS = int(get_hp("target_steps", "1500"))
SAVE_EVERY = int(get_hp("save_every", "100"))
RANK = int(get_hp("rank", "4"))
ALPHA = int(get_hp("alpha", str(RANK)))
LR = float(get_hp("lr", "4e-4"))

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)

token = os.environ.get("HF_TOKEN") or HPS.get("hf_token") or ""
if token:
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

def run_env() -> dict[str, str]:
    env = os.environ.copy()
    extra = os.pathsep.join([str(THIS_DIR), str(TOOLKIT_DIR)])
    env["PYTHONPATH"] = extra + os.pathsep + env.get("PYTHONPATH", "")
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env

def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, env=run_env())

def install_schema_compat() -> None:
    import site
    import schema_compat

    schema_compat.patch_toolkit_files(TOOLKIT_DIR)
    sp = Path(site.getsitepackages()[0])
    shutil.copy2(THIS_DIR / "schema_compat.py", sp / "schema_compat.py")
    sc = sp / "sitecustomize.py"
    snippet = "\nimport schema_compat\nschema_compat.apply()\n"
    if sc.exists():
        text = sc.read_text(encoding="utf-8")
        if "schema_compat" not in text:
            sc.write_text(text + snippet, encoding="utf-8")
    else:
        sc.write_text(snippet, encoding="utf-8")
    schema_compat.apply()

def install_toolkit() -> None:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    torch_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "torch>=2.6.0",
        "torchvision",
        "torchaudio",
        "--index-url",
        "https://download.pytorch.org/whl/cu124",
    ]
    run(torch_cmd)
    if not TOOLKIT_DIR.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/ostris/ai-toolkit.git", str(TOOLKIT_DIR)])
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=TOOLKIT_DIR)
    run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"], cwd=TOOLKIT_DIR)
    run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "lycoris_lora", "bitsandbytes"])
    run(torch_cmd)
    install_schema_compat()

def seed_checkpoints() -> None:
    import re
    run_dir = OUTPUT_DIR / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove any existing .sagemaker-uploaded or temporary marker files
    for marker in CKPT_DIR.rglob("*.sagemaker-uploaded"):
        try:
            marker.unlink()
            print(f"Removed stale marker: {marker.name}", flush=True)
        except Exception:
            pass

    if RESUME_STEP == 0:
        print("Starting training from scratch (RESUME_STEP=0). Skipping checkpoint seeding.", flush=True)
        return

    print(f"Seeding checkpoints from {SEED_DIR} to {run_dir}...", flush=True)
    if SEED_DIR.exists():
        for item in sorted(SEED_DIR.iterdir()):
            if item.is_file() and not item.name.endswith(".sagemaker-uploaded"):
                dest = run_dir / item.name
                shutil.copy2(item, dest)
                print(f"  seeded: {item.name} ({item.stat().st_size} bytes)", flush=True)
                
                # Create aliases matching RUN_NAME so ai-toolkit get_latest_save_path finds it
                if item.suffix == ".safetensors":
                    step_match = re.search(r"_(\d{9})\.safetensors$", item.name)
                    if step_match:
                        step_val = int(step_match.group(1))
                        alias = run_dir / f"{RUN_NAME}_{step_val:09d}.safetensors"
                        shutil.copy2(item, alias)
                        print(f"  aliased: {alias.name}", flush=True)
                    # Also create alias for target RESUME_STEP if not already present
                    resume_alias = run_dir / f"{RUN_NAME}_{RESUME_STEP:09d}.safetensors"
                    if not resume_alias.exists():
                        shutil.copy2(item, resume_alias)
                        print(f"  aliased: {resume_alias.name}", flush=True)
                    # Default un-stepped alias
                    unstepped = run_dir / f"{RUN_NAME}.safetensors"
                    if not unstepped.exists():
                        shutil.copy2(item, unstepped)

    # List seeded files
    print(f"Contents of {run_dir}:", flush=True)
    for p in sorted(run_dir.glob("*")):
        print(f"  {p.name} ({p.stat().st_size} bytes)", flush=True)

def write_config() -> Path:
    config = {
        "job": "extension",
        "config": {
            "name": RUN_NAME,
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": str(OUTPUT_DIR),
                    "device": "cuda:0",
                    "network": {
                        "type": ADAPTER_TYPE,
                        "linear": RANK,
                        "linear_alpha": ALPHA,
                    },
                    "save": {
                        "dtype": "bfloat16",
                        "save_every": SAVE_EVERY,
                        "max_step_saves_to_keep": 25,
                        "push_to_hub": False,
                    },
                    "datasets": [
                        {
                            "folder_path": str(TRAINING),
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "shuffle_tokens": False,
                            "cache_latents_to_disk": True,
                            "resolution": [512, 768, 1024],
                        }
                    ],
                    "train": {
                        "batch_size": 1,
                        "steps": TARGET_STEPS,
                        "start_step": RESUME_STEP,
                        "gradient_accumulation_steps": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw",
                        "lr": LR,
                        "dtype": "bf16",
                    },
                    "model": {
                        "name_or_path": "black-forest-labs/FLUX.1-dev",
                        "is_flux": True,
                        "quantize": True,
                        "quantize_te": True,
                        "low_vram": True,
                    },
                }
            ],
        },
    }

    import yaml
    cfg_file = CKPT_DIR / "config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    print(f"Wrote ai-toolkit config to {cfg_file}", flush=True)
    return cfg_file

def copy_to_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUTPUT_DIR / RUN_NAME
    if run_dir.exists():
        dest = MODEL_DIR / RUN_NAME
        dest.mkdir(parents=True, exist_ok=True)
        for p in sorted(run_dir.iterdir()):
            if p.suffix in {".safetensors", ".pt", ".yaml"}:
                shutil.copy2(p, dest / p.name)
                print(f"Saved to model output: {p.name}", flush=True)

def main() -> None:
    print(f"=== Starting SageMaker FLUX Training for {ADAPTER_TYPE.upper()} ===", flush=True)
    print(f"Run name: {RUN_NAME}, Target Steps: {TARGET_STEPS}, Resume Step: {RESUME_STEP}", flush=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    seed_checkpoints()
    install_toolkit()
    cfg_path = write_config()

    cmd = [sys.executable, "run.py", str(cfg_path)]
    print(f"Executing: {' '.join(cmd)} in {TOOLKIT_DIR}", flush=True)
    run(cmd, cwd=TOOLKIT_DIR)

    copy_to_model_dir()
    print("=== Training Job Completed Successfully ===", flush=True)

if __name__ == "__main__":
    main()
