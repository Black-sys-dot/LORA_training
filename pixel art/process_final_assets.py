import cv2
import numpy as np
from PIL import Image
import os

def process_image(input_path, output_path):
    print(f"Processing {input_path}...")
    
    # 1. LOAD IMAGE
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)
    
    # 2. CHROMA KEY (Remove Pure Neon Cyan)
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    cyan_mask = (g.astype(int) > 200) & (b.astype(int) > 200) & (r.astype(int) < 100)
    data[cyan_mask, 3] = 0
    
    # 3. ERODE (Shave the 2-pixel anti-aliased outer edge)
    alpha = data[:, :, 3]
    kernel = np.ones((3,3), np.uint8)
    eroded_alpha = cv2.erode(alpha, kernel, iterations=2)
    data[:, :, 3] = eroded_alpha
    
    # 4. MIXEL SNAP (1024 -> 256 -> 1024 via Nearest Neighbor)
    temp_pil = Image.fromarray(data)
    temp_pil = temp_pil.resize((256, 256), Image.Resampling.NEAREST)
    temp_pil = temp_pil.resize((1024, 1024), Image.Resampling.NEAREST)
    data = np.array(temp_pil)
    
    # 5. PALETTE QUANTIZATION (16 Colors, strictly on the character body)
    valid_pixels = data[:, :, 3] > 0
    rgb_values = data[valid_pixels, :3].astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 16
    _, labels, centers = cv2.kmeans(rgb_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    quantized_rgb = centers[labels.flatten()]
    data[valid_pixels, :3] = quantized_rgb
    
    # 6. CROP & SAVE (Transparent PNG)
    non_empty_columns = np.where(data[:, :, 3].max(axis=0) > 0)[0]
    non_empty_rows = np.where(data[:, :, 3].max(axis=1) > 0)[0]
    if len(non_empty_columns) > 0 and len(non_empty_rows) > 0:
        crop_box = (min(non_empty_columns), min(non_empty_rows), max(non_empty_columns), max(non_empty_rows))
        final_pil = Image.fromarray(data).crop(crop_box)
    else:
        final_pil = Image.fromarray(data)
        
    final_pil.save(output_path, "PNG")
    print(f"Saved finished asset to {output_path}")

def main():
    os.makedirs("../results/final_assets", exist_ok=True)
    
    # Process Seed 7 (User's Manual Edit)
    process_image(
        "/home/black/Downloads/blue_girl_seed_7.png", 
        "../results/final_assets/final_asset_seed_7.png"
    )
    
    # Process Seed 8 (Artistic Choice)
    process_image(
        "../results/batch_test/blue_girl_seed_8.png", 
        "../results/final_assets/final_asset_seed_8.png"
    )
    print("\nAll assets successfully processed!")

if __name__ == "__main__":
    main()
