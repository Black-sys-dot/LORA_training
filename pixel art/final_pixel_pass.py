import numpy as np
from PIL import Image, ImageChops
import os
from sklearn.cluster import MiniBatchKMeans

def apply_final_pixel_pass(input_path, output_path, offset=2, colors=32):
    print(f"Applying final pixel art pass to {input_path}...")
    
    img = Image.open(input_path).convert("RGBA")
    
    # 1. GRID PHASE OFFSET
    # Shift the image to perfectly align the AI's internal grid with the mathematical grid
    shifted_img = ImageChops.offset(img, offset, offset)
    
    # 2. MIXEL SNAP
    # Downscale to 256x256 to lock into a Gameboy-style retro grid, then upscale
    snapped_img = shifted_img.resize((256, 256), Image.Resampling.NEAREST)
    snapped_img = snapped_img.resize((1024, 1024), Image.Resampling.NEAREST)
    
    # Revert the shift
    final_img = ImageChops.offset(snapped_img, -offset, -offset)
    
    data = np.array(final_img)
    alpha = data[:, :, 3]
    
    # 3. PALETTE QUANTIZATION
    # Flatten the digital AI gradients into exactly 32 solid retro colors
    mask = alpha > 0
    visible_pixels = data[mask][:, :3] 
    
    kmeans = MiniBatchKMeans(n_clusters=colors, random_state=42, n_init="auto")
    kmeans.fit(visible_pixels)
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    data[mask, :3] = quantized_pixels
    
    # Save the absolute final asset
    final_pil = Image.fromarray(data)
    final_pil.save(output_path, "PNG")
    print(f"Absolute final asset saved to {output_path}")

def main():
    os.makedirs("/home/black/Projects/LORA_training/results/final_assets", exist_ok=True)
    apply_final_pixel_pass(
        "/home/black/Projects/LORA_training/results/final_assets/cozy_girl_hybrid_clean.png", 
        "/home/black/Projects/LORA_training/results/final_assets/cozy_girl_seed_6_DONE.png"
    )

if __name__ == "__main__":
    main()
