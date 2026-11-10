import cv2
import numpy as np
from PIL import Image, ImageChops
import os
from sklearn.cluster import MiniBatchKMeans

def crisp_solid_pipeline(input_path, output_path, offset=2, colors=32):
    print(f"Applying Crisp Edge Solid Pipeline to {input_path}...")
    
    img = Image.open(input_path).convert("RGB")
    
    # 1. GRID PHASE OFFSET & MIXEL SNAP
    shifted_img = ImageChops.offset(img, offset, offset)
    snapped_img = shifted_img.resize((256, 256), Image.Resampling.NEAREST)
    snapped_img = snapped_img.resize((1024, 1024), Image.Resampling.NEAREST)
    final_img = ImageChops.offset(snapped_img, -offset, -offset)
    
    data = np.array(final_img)
    r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
    
    # 2. BULLETPROOF BACKGROUND DETECTION
    raw_bg_mask = (g > 200) & (b > 200) & (r < 150)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_bg_mask.astype(np.uint8), connectivity=4)
    
    largest_cyan_label = 1
    max_area = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            largest_cyan_label = i
            
    true_bg_mask = (labels == largest_cyan_label).astype(np.uint8)
    
    # 3. DILATE THE BACKGROUND TO EAT THE BLUR
    # By expanding the background mask by a few pixels, we mathematically swallow
    # the soft, blurry anti-aliased edge, leaving only the sharp, solid inner colors of the character!
    kernel = np.ones((5, 5), np.uint8)
    dilated_bg_mask = cv2.dilate(true_bg_mask, kernel, iterations=1)
    
    # The character is everything that isn't the expanded background
    char_mask = (dilated_bg_mask == 0)
    
    # 4. K-MEANS PALETTE (TRAINED ONLY ON THE CRISP CHARACTER)
    char_pixels = data[char_mask]
    
    kmeans = MiniBatchKMeans(n_clusters=colors, random_state=42, n_init="auto")
    kmeans.fit(char_pixels)
    quantized_char_pixels = kmeans.cluster_centers_[kmeans.labels_]
    
    # 5. APPLY THE CRISP PALETTE
    data[char_mask] = quantized_char_pixels
    
    # 6. FLATTEN THE BACKGROUND (This fills in the eaten blur pixels with solid background)
    data[dilated_bg_mask == 1] = [112, 254, 255]
    
    final_pil = Image.fromarray(data.astype(np.uint8))
    final_pil.save(output_path, "PNG")
    print(f"Crisp solid background asset saved to {output_path}")

def main():
    os.makedirs("/home/black/Projects/LORA_training/results/final_assets", exist_ok=True)
    crisp_solid_pipeline(
        "/home/black/Downloads/blue_girl_seed_7.png", 
        "/home/black/Projects/LORA_training/results/final_assets/blue_girl_seed_7_CRISP.png"
    )

if __name__ == "__main__":
    main()
