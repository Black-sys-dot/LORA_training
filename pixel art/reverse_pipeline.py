import cv2
import numpy as np
from PIL import Image, ImageChops
import os
from sklearn.cluster import MiniBatchKMeans

def reverse_pipeline(input_path, output_path, offset=2, colors=32):
    print(f"Applying REVERSE Pipeline to {input_path}...")
    
    img = Image.open(input_path).convert("RGB")
    data = np.array(img)
    r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
    
    # 1. BULLETPROOF BACKGROUND DETECTION
    raw_bg_mask = (g > 200) & (b > 200) & (r < 150)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_bg_mask.astype(np.uint8), connectivity=4)
    
    largest_cyan_label = 1
    max_area = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            largest_cyan_label = i
            
    true_bg_mask = (labels == largest_cyan_label)
    
    # 2. FLATTEN THE BACKGROUND FIRST (1 color slot)
    data[true_bg_mask] = [112, 254, 255]
    
    # 3. K-MEANS PALETTE (ON THE RAW 1024x1024 IMAGE)
    # This forces the microscopically thin anti-aliasing to snap to hard colors BEFORE downscaling!
    h, w, c = data.shape
    all_pixels = data.reshape(-1, 3)
    
    kmeans = MiniBatchKMeans(n_clusters=colors, random_state=42, n_init="auto")
    kmeans.fit(all_pixels)
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    
    quantized_data = quantized_pixels.reshape(h, w, 3).astype(np.uint8)
    
    # 4. MIXEL SNAP (AFTER K-MEANS)
    quantized_pil = Image.fromarray(quantized_data)
    shifted_img = ImageChops.offset(quantized_pil, offset, offset)
    snapped_img = shifted_img.resize((256, 256), Image.Resampling.NEAREST)
    snapped_img = snapped_img.resize((1024, 1024), Image.Resampling.NEAREST)
    final_img = ImageChops.offset(snapped_img, -offset, -offset)
    
    final_img.save(output_path, "PNG")
    print(f"Reverse pipeline asset saved to {output_path}")

def main():
    os.makedirs("/home/black/Projects/LORA_training/results/final_assets", exist_ok=True)
    reverse_pipeline(
        "/home/black/Downloads/blue_girl_seed_7.png", 
        "/home/black/Projects/LORA_training/results/final_assets/blue_girl_seed_7_REVERSE.png"
    )

if __name__ == "__main__":
    main()
