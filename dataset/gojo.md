# Gojo FLUX.1 LoRA Training Project - Master Context File

## 🎯 Project Overview
We successfully trained a high-quality LoRA on the **FLUX.1-dev** 12-Billion parameter base model using a 25-image dataset of Satoru Gojo. This file serves as the master record of our setup, bugs encountered, and hyperparameter experiments so we can replicate this success for future datasets.

## 📁 Critical Folder Paths
- **Working Directory:** `/home/black/Projects/LORA_training`
- **Virtual Environment:** `/home/black/Projects/LORA_training/venv` (Must be activated to run Modal commands)
- **Dataset Folder:** `/home/black/Projects/LORA_training/dataset/gojo`
- **Training Scripts:** `/home/black/Projects/LORA_training/codes/` (`train.py`, `train_rank32.py`, `train_slowbake.py`)
- **Inference Scripts:** `/home/black/Projects/LORA_training/codes/` (`infer.py`, `test_experiments.py`, `test_attacks.py`)
- **Generated Images:** `/home/black/Projects/LORA_training/results/`
- **Local Checkpoint Backups:** `/home/black/Projects/LORA_training/checkpoints/`
- **Master Archive Backup:** `/home/black/LoRA_Backups/Gojo_FLUX_LoRA_Backup_Archive.tar.gz`

## ☁️ Cloud Infrastructure (Modal)
- **Framework:** `ai-toolkit` running dynamically inside Modal ephemeral containers.
- **Volume:** `flux-lora-volume` (This is permanent cloud storage where the dataset is uploaded and checkpoints are saved. Mounted to `/data` in the container).
- **Secrets:** `huggingface-secret` (Required to download the base `FLUX.1-dev` model).

## 🐛 Critical Bugs & Fixes (READ BEFORE NEW RUNS)
1. **The H100 MMU Hardware Crash:** 
   - **Bug:** Training FLUX.1 on H100 GPUs using PyTorch 2.3/2.4 with bfloat16 caused a deep silicon-level `MMU Fault: ENGINE GRAPHICS... ACCESS_TYPE_VIRT_READ` crash during the backward pass.
   - **Fix:** We completely abandoned Hopper GPUs and switched the Modal request to **A100-80GB**. The Ampere architecture bypasses this bug entirely. **Always use A100-80GB for training.**
2. **Inference Un-Fused LoRA Bug:**
   - **Bug:** `diffusers` would load the LoRA weights into memory but ignore them during generation, resulting in base-model images. It also threw an error requiring `peft`.
   - **Fix:** You MUST `pip install peft` in the Modal inference container. Furthermore, you must explicitly call `pipe.fuse_lora(lora_scale=1.0)` after `load_lora_weights()` to force PyTorch to inject the weights into the FLUX model.
3. **Modal Network vs. Eventlet:**
   - **Bug:** If using Flask-SocketIO locally to monitor logs, `eventlet.monkey_patch()` breaks the Modal CLI's async transport.
   - **Fix:** Use standard threading (`async_mode='threading'`) for local UI dashboards instead of eventlet.

## 🧪 Hyperparameter Sweeps & Findings
We ran three different experiments to find the optimal bake. **The very first run was the undeniable champion.**
1. **The Original Masterpiece (WINNER):**
   - **Config:** Rank/Linear 16, Learning Rate `4e-4`, Optimizer `adamw`, 2000 steps.
   - **Result:** Absolute perfection. Fully learned the likeness but remained flexible enough to adapt to diverse styles and environments.
2. **Rank 32 Sweep (Overfit):**
   - **Config:** Rank/Linear 32, 2000 steps.
   - **Result:** Giving the AI a "bigger brain" did not improve the likeness. Instead, it overfitted to the 25 images and caused anatomical instability (e.g., drawing 3 hands) during dynamic action poses.
3. **Slow Bake Sweep (Timeout):**
   - **Config:** Learning Rate `1e-4`, 4000 steps.
   - **Result:** Hit Modal's strict 2-hour server limit at step 3947 and was killed. We recovered the Step 3000 checkpoint via `save_every: 1000`. Good quality, but not superior to the original 2000-step Rank 16 run.

## 🗣️ Prompting Mechanics
Because the dataset captions were meticulously detailed, the LoRA tied Gojo's likeness to specific descriptive phrases rather than just the word "Gojo".
- **Bad Prompt:** `"Satoru Gojo drinking tea."` (Result: Random white-haired anime guy).
- **Good Prompt:** `"Satoru Gojo, adult male, pure-white hair, icy-blue eyes, dark navy-black high-collared Jujutsu Kaisen-style uniform..."` (Result: Flawless character generation).
- **Lesson:** Always open the dataset `.txt` files and extract the exact trigger tags to use during inference.

---
*End of Context File. If initializing a new dataset, replicate the `train.py` configuration (A100-80GB, Rank 16, LR 4e-4, 2000 steps).*
