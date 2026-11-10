# Pixel Art FLUX.1 LoRA Training

## 🎯 Goal
Train a clean, dithering-free, cel-shaded Pixel Art style LoRA based on 51 high-quality images.

## 📁 Paths
- **Dataset:** `/home/black/Projects/LORA_training/pixel art/dataset`
- **Training Script:** `/home/black/Projects/LORA_training/pixel art/train.py`
- **Inference Script:** `/home/black/Projects/LORA_training/pixel art/test_pixel.py`
- **Modal Volume:** `flux-lora-volume` (Dataset uploaded to `/dataset/pixel_art/dataset`)

## 📊 Run 1 (Interrupted)
- **Config:** Rank 32, LR 4e-4, 4000 steps target.
- **Outcome:** The local laptop disconnected, causing Modal to terminate the ephemeral app at **Step 3,307**.
- **Saved Assets:** The Step 1000, 2000, and 3000 `.safetensors` files were safely written to the Modal volume, alongside the massive 1.3 GiB `optimizer.pt` file.
- **Local Backups:** All checkpoints have been downloaded to `LORA_training/checkpoints/`.
- **Test Result:** The Step 3000 model was tested and produced "crazy good" results.

## 🔜 Next Steps (Tomorrow)
- Resume the training from Step 3000 using the `optimizer.pt` in the cloud volume to finish the final 1000 steps.
- To do this, we will need to update `train.py` to point to the `pixel_art_lora` folder in the volume and set it to resume so it loads the optimizer state.
