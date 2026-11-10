import cv2
import numpy as np
from PIL import Image, ImageChops
import os
from sklearn.cluster import MiniBatchKMeans

def true_solid_pipeline_v3(input_path, output_path, offset=2, colors=32):
    print(f"Applying Bulletproof Solid Pipeline v3 to {input_path}...")
    
    img = Image.open(input_path).convert("RGB")
    
    shifted_img = ImageChops.offset(img, offset, offset)
    snapped_img = shifted_img.resize((256, 256), Image.Resampling.NEAREST)
    snapped_img = snapped_img.resize((1024, 1024), Image.Resampling.NEAREST)
    final_img = ImageChops.offset(snapped_img, -offset, -offset)
    
    data = np.array(final_img)
    r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
    
    # 2. BULLETPROOF BACKGROUND DETECTION
    # Lowered the threshold slightly to catch subtle vignettes in the corners
    raw_bg_mask = (g > 200) & (b > 200) & (r < 150)
    
    # Label 0 is the character. Labels 1+ are the background.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_bg_mask.astype(np.uint8), connectivity=4)
    
    # Find the largest cyan component (the actual background)
    largest_cyan_label = 1
    max_area = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            largest_cyan_label = i
            
    true_bg_mask = (labels == largest_cyan_label)
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
    true_solid_pipeline_v3(
        "/home/black/Downloads/blue_girl_seed_7.png", 
        "/home/black/Projects/LORA_training/results/final_assets/blue_girl_seed_7_PERFECT.png"
    )

if __name__ == "__main__":
    main()
