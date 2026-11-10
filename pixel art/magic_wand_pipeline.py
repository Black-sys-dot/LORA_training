import cv2
import numpy as np
from PIL import Image
import os

def magic_wand_chroma_key(img_path, output_path):
    print(f"Processing {img_path} with Magic Wand...")
    
    # 1. LOAD IMAGE (Strictly 3-channel BGR for floodFill)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # 2. FLOOD FILL
    temp_bg_color = (255, 0, 255) # Magenta in BGR
    # Start at (10,10) for the main sky/background
    cv2.floodFill(img, mask, (10, 10), temp_bg_color, (10, 10, 10), (10, 10, 10), flags=4)
    # Start at (512, 1010) to delete the background trapped between her legs!
    cv2.floodFill(img, mask, (512, 1010), temp_bg_color, (10, 10, 10), (10, 10, 10), flags=4)
    
    # Create the RGBA image
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    
    # Set Magenta to transparent
    bg_mask = (img_rgba[:, :, 0] == 255) & (img_rgba[:, :, 1] == 0) & (img_rgba[:, :, 2] == 255)
    img_rgba[bg_mask, 3] = 0
    
    data = img_rgba
    
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
    
    # 5. PALETTE QUANTIZATION (16 Colors)
    valid_pixels = data[:, :, 3] > 0
    rgb_values = data[valid_pixels, :3].astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 16
    _, labels, centers = cv2.kmeans(rgb_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    quantized_rgb = centers[labels.flatten()]
    data[valid_pixels, :3] = quantized_rgb
    
    # 6. CROP & SAVE
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
    magic_wand_chroma_key(
        "/home/black/Downloads/blue_girl_seed_7.png", 
        "../results/final_assets/final_asset_seed_7_fixed.png"
    )
    
    # Process Seed 8
    magic_wand_chroma_key(
        "../results/batch_test/blue_girl_seed_8.png", 
        "../results/final_assets/final_asset_seed_8_fixed.png"
    )
    print("\nAll assets successfully processed with Magic Wand!")

if __name__ == "__main__":
    main()
