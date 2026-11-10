import cv2
import numpy as np
from PIL import Image, ImageChops
import os
from sklearn.cluster import MiniBatchKMeans

def apply_offset_pipeline(input_path, output_path, offset_x, offset_y, colors=32):
    print(f"Applying perfect grid offset to {input_path}...")
    
    img = Image.open(input_path).convert("RGBA")
    
    shifted_img = ImageChops.offset(img, offset_x, offset_y)
    
    snapped_img = shifted_img.resize((256, 256), Image.Resampling.NEAREST)
    snapped_img = snapped_img.resize((1024, 1024), Image.Resampling.NEAREST)
    
    final_img = ImageChops.offset(snapped_img, -offset_x, -offset_y)
    
    data = np.array(final_img)
    alpha = data[:, :, 3]
    
    mask = alpha > 0
    visible_pixels = data[mask][:, :3] 
    
    kmeans = MiniBatchKMeans(n_clusters=colors, random_state=42, n_init="auto")
    kmeans.fit(visible_pixels)
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    data[mask, :3] = quantized_pixels
    
    final_pil = Image.fromarray(data)
    final_pil.save(output_path, "PNG")
    print(f"Saved successfully to {output_path}")

def main():
    os.makedirs("../results/final_assets/grid_tests", exist_ok=True)
    input_file = "../results/final_assets/cozy_girl_ULTIMATE_7.png"
    out_name = "../results/final_assets/grid_tests/cozy_girl_offset_2_2_seed_7.png"
    
    # Apply the magic (2,2) offset to Seed 7
    apply_offset_pipeline(input_file, out_name, 2, 2)

if __name__ == "__main__":
    main()
