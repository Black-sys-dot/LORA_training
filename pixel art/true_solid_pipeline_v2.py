import cv2
import numpy as np
from PIL import Image, ImageChops
import os
from sklearn.cluster import MiniBatchKMeans

def true_solid_pipeline_v2(input_path, output_path, offset=2, colors=32):
    print(f"Applying Bulletproof Solid Pipeline to {input_path}...")
    
    img = Image.open(input_path).convert("RGB")
    
    # 1. GRID PHASE OFFSET & MIXEL SNAP
    shifted_img = ImageChops.offset(img, offset, offset)
    snapped_img = shifted_img.resize((256, 256), Image.Resampling.NEAREST)
    snapped_img = snapped_img.resize((1024, 1024), Image.Resampling.NEAREST)
    final_img = ImageChops.offset(snapped_img, -offset, -offset)
    
    data = np.array(final_img)
    r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
    
    # 2. BULLETPROOF BACKGROUND DETECTION
    raw_bg_mask = (g > 240) & (b > 240) & (r < 150)
    
    # Run Connected Components to find the actual background (the one touching 0,0)
    # This prevents cyan highlights inside her hair from being mistaken for background!
    num_labels, labels = cv2.connectedComponents(raw_bg_mask.astype(np.uint8), connectivity=4)
    bg_label = labels[0, 0] # The background is whatever touches the top-left corner
    
    true_bg_mask = (labels == bg_label)
    char_mask = ~true_bg_mask
    
    # 3. K-MEANS PALETTE (TRAINED ONLY ON THE CHARACTER)
    char_pixels = data[char_mask]
    
    kmeans = MiniBatchKMeans(n_clusters=colors, random_state=42, n_init="auto")
    kmeans.fit(char_pixels)
    quantized_char_pixels = kmeans.cluster_centers_[kmeans.labels_]
    
    # 4. APPLY THE PALETTE
    data[char_mask] = quantized_char_pixels
    
    # 5. FLATTEN THE BACKGROUND
    data[true_bg_mask] = [112, 254, 255]
    
    final_pil = Image.fromarray(data.astype(np.uint8))
    final_pil.save(output_path, "PNG")
    print(f"Bulletproof solid background asset saved to {output_path}")

def main():
    os.makedirs("/home/black/Projects/LORA_training/results/final_assets", exist_ok=True)
    true_solid_pipeline_v2(
        "/home/black/Downloads/blue_girl_seed_7.png", 
        "/home/black/Projects/LORA_training/results/final_assets/blue_girl_seed_7_PERFECT.png"
    )

if __name__ == "__main__":
    main()
