import numpy as np
from PIL import Image, ImageChops
import os
from sklearn.cluster import MiniBatchKMeans

def solid_bg_pipeline(input_path, output_path, offset=2, colors=32):
    print(f"Applying pure pixel art pipeline to {input_path} (Keeping background)...")
    
    img = Image.open(input_path).convert("RGB")
    
    # 1. GRID PHASE OFFSET
    shifted_img = ImageChops.offset(img, offset, offset)
    
    # 2. MIXEL SNAP
    snapped_img = shifted_img.resize((256, 256), Image.Resampling.NEAREST)
    snapped_img = snapped_img.resize((1024, 1024), Image.Resampling.NEAREST)
    
    # Revert the shift
    final_img = ImageChops.offset(snapped_img, -offset, -offset)
    
    data = np.array(final_img)
    
    # 3. K-MEANS PALETTE (Applied to the entire image, including background)
    h, w, c = data.shape
    pixels = data.reshape(-1, 3)
    
    kmeans = MiniBatchKMeans(n_clusters=colors, random_state=42, n_init="auto")
    kmeans.fit(pixels)
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    
    final_data = quantized_pixels.reshape(h, w, 3).astype(np.uint8)
    
    # Save the absolute final asset
    final_pil = Image.fromarray(final_data)
    final_pil.save(output_path, "PNG")
    print(f"Solid background masterpiece saved to {output_path}")

def main():
    os.makedirs("/home/black/Projects/LORA_training/results/final_assets", exist_ok=True)
    solid_bg_pipeline(
        "/home/black/Projects/LORA_training/results/batch_test/hands_test/hands_seed_6.png", 
        "/home/black/Projects/LORA_training/results/final_assets/cozy_girl_seed_6_SOLID.png"
    )

if __name__ == "__main__":
    main()
