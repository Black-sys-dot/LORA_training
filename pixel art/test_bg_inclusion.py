import cv2
import numpy as np
from PIL import Image, ImageChops
import os
from sklearn.cluster import MiniBatchKMeans

def test_bg_inclusion(input_path, output_path, offset=2, colors=32):
    print(f"Applying Background-Included Pipeline to {input_path}...")
    
    img = Image.open(input_path).convert("RGB")
    
    shifted_img = ImageChops.offset(img, offset, offset)
    snapped_img = shifted_img.resize((256, 256), Image.Resampling.NEAREST)
    snapped_img = snapped_img.resize((1024, 1024), Image.Resampling.NEAREST)
    final_img = ImageChops.offset(snapped_img, -offset, -offset)
    
    data = np.array(final_img)
    r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
    
    # Detect the background
    raw_bg_mask = (g > 200) & (b > 200) & (r < 150)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_bg_mask.astype(np.uint8), connectivity=4)
    
    largest_cyan_label = 1
    max_area = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            largest_cyan_label = i
            
    true_bg_mask = (labels == largest_cyan_label)
    
    # 1. FLATTEN THE BACKGROUND FIRST
    # By forcing the background to be exactly one mathematical color before running K-Means, 
    # we guarantee K-Means will only spend 1 palette slot on it (an infinitely dense cluster)
    data[true_bg_mask] = [112, 254, 255]
    
    # 2. RUN K-MEANS ON THE ENTIRE IMAGE
    # The character gets 31 colors, the background gets 1 color.
    # The anti-aliased edges will cluster into whichever is closer.
    h, w, c = data.shape
    all_pixels = data.reshape(-1, 3)
    
    kmeans = MiniBatchKMeans(n_clusters=colors, random_state=42, n_init="auto")
    kmeans.fit(all_pixels)
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    
    final_data = quantized_pixels.reshape(h, w, 3).astype(np.uint8)
    
    final_pil = Image.fromarray(final_data)
    final_pil.save(output_path, "PNG")
    print(f"Test asset saved to {output_path}")

def main():
    os.makedirs("/home/black/Projects/LORA_training/results/final_assets", exist_ok=True)
    test_bg_inclusion(
        "/home/black/Downloads/blue_girl_seed_7.png", 
        "/home/black/Projects/LORA_training/results/final_assets/blue_girl_seed_7_INCLUDED.png"
    )

if __name__ == "__main__":
    main()
