import cv2
import numpy as np
from PIL import Image

def run_pipeline():
    # 1. LOAD IMAGE
    img_path = "../results/batch_test/blue_girl_seed_7_fixed.png"
    print("Loading image...")
    img = Image.open(img_path).convert("RGBA")
    data = np.array(img)
    
    # 2. CHROMA KEY (Remove Cyan)
    print("Running Chroma Key...")
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    # NEON CYAN is extremely bright green and blue, and very low red.
    # We must be strict so we don't accidentally erase her blue jeans or blue hair!
    cyan_mask = (g.astype(int) > 200) & (b.astype(int) > 200) & (r.astype(int) < 100)
    data[cyan_mask, 3] = 0
    
    # 3. ERODE (The "Black Border Removal" trick to shave the 2-pixel outer edge)
    print("Shaving outer edge...")
    alpha = data[:, :, 3]
    kernel = np.ones((3,3), np.uint8)
    eroded_alpha = cv2.erode(alpha, kernel, iterations=2)
    data[:, :, 3] = eroded_alpha
    
    # 4. MIXEL SNAP (1024 -> 256 -> 1024 via Nearest Neighbor)
    print("Snapping Mixels to Grid...")
    temp_pil = Image.fromarray(data)
    temp_pil = temp_pil.resize((256, 256), Image.Resampling.NEAREST)
    temp_pil = temp_pil.resize((1024, 1024), Image.Resampling.NEAREST)
    data = np.array(temp_pil)
    
    # 5. PALETTE QUANTIZATION (16 Colors, ONLY applied to the girl!)
    print("Quantizing Palette (16 Colors)...")
    valid_pixels = data[:, :, 3] > 0
    rgb_values = data[valid_pixels, :3].astype(np.float32)
    
    # Use K-Means clustering to find the 16 most dominant colors IN THE GIRL
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 16
    _, labels, (centers) = cv2.kmeans(rgb_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Snap her pixels to the 16 colors
    centers = np.uint8(centers)
    quantized_rgb = centers[labels.flatten()]
    data[valid_pixels, :3] = quantized_rgb
    
    # 6. CROP & SAVE (NO BACKGROUND)
    print("Cropping and saving...")
    non_empty_columns = np.where(data[:, :, 3].max(axis=0) > 0)[0]
    non_empty_rows = np.where(data[:, :, 3].max(axis=1) > 0)[0]
    if len(non_empty_columns) > 0 and len(non_empty_rows) > 0:
        crop_box = (min(non_empty_columns), min(non_empty_rows), max(non_empty_columns), max(non_empty_rows))
        final_pil = Image.fromarray(data).crop(crop_box)
    else:
        final_pil = Image.fromarray(data)
        
    save_path = "../results/smart_greenscreen/7_final_pipeline_girl_only.png"
    final_pil.save(save_path, "PNG")
    print(f"Done! Saved to {save_path}")

if __name__ == "__main__":
    run_pipeline()
