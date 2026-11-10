import cv2
import numpy as np
from PIL import Image, ImageChops
import os
from sklearn.cluster import MiniBatchKMeans

def save_256_versions(input_path, out_before, out_after, offset=2, colors=32):
    print("Generating native 256x256 versions...")
    
    img = Image.open(input_path).convert("RGB")
    
    # 1. APPLY MIXEL SNAP (No Palette Change Yet)
    shifted_img = ImageChops.offset(img, offset, offset)
    snapped_img = shifted_img.resize((256, 256), Image.Resampling.NEAREST)
    snapped_img_1024 = snapped_img.resize((1024, 1024), Image.Resampling.NEAREST)
    img_before_palette = ImageChops.offset(snapped_img_1024, -offset, -offset)
    
    # Save the 256x256 "Before Palette" version
    img_before_palette.resize((256, 256), Image.Resampling.NEAREST).save(out_before, "PNG")
    print(f"Saved Before Palette: {out_before}")
    
    # 2. APPLY K-MEANS PALETTE
    data = np.array(img_before_palette)
    r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
    
    # Detect background
    raw_bg_mask = (g > 200) & (b > 200) & (r < 150)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_bg_mask.astype(np.uint8), connectivity=4)
    
    largest_cyan_label = 1
    max_area = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            largest_cyan_label = i
            
    true_bg_mask = (labels == largest_cyan_label)
    
    # Flatten background
    data[true_bg_mask] = [112, 254, 255]
    
    # Run K-Means on full image
    h, w, c = data.shape
    all_pixels = data.reshape(-1, 3)
    
    kmeans = MiniBatchKMeans(n_clusters=colors, random_state=42, n_init="auto")
    kmeans.fit(all_pixels)
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    
    final_data = quantized_pixels.reshape(h, w, 3).astype(np.uint8)
    final_pil = Image.fromarray(final_data)
    
    # Save the 256x256 "After Palette" version
    final_pil.resize((256, 256), Image.Resampling.NEAREST).save(out_after, "PNG")
    print(f"Saved After Palette: {out_after}")

def main():
    os.makedirs("/home/black/Projects/LORA_training/results/final_assets", exist_ok=True)
    save_256_versions(
        "/home/black/Downloads/blue_girl_seed_7.png", 
        "/home/black/Projects/LORA_training/results/final_assets/blue_girl_256_BEFORE.png",
        "/home/black/Projects/LORA_training/results/final_assets/blue_girl_256_AFTER.png"
    )

if __name__ == "__main__":
    main()
