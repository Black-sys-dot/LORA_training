import cv2
import numpy as np
from PIL import Image, ImageChops
import os
from sklearn.cluster import MiniBatchKMeans

def apply_offset_pipeline(input_path, output_path, offset_x, offset_y, colors=32):
    print(f"Testing offset X:{offset_x} Y:{offset_y} on {input_path}...")
    
    img = Image.open(input_path).convert("RGBA")
    
    # SHIFT THE IMAGE BY THE OFFSET
    # We use ImageChops.offset to seamlessly shift the image by a few pixels
    shifted_img = ImageChops.offset(img, offset_x, offset_y)
    
    # MIXEL SNAP
    # Downscale and upscale on the shifted grid
    snapped_img = shifted_img.resize((256, 256), Image.Resampling.NEAREST)
    snapped_img = snapped_img.resize((1024, 1024), Image.Resampling.NEAREST)
    
    # SHIFT IT BACK
    final_img = ImageChops.offset(snapped_img, -offset_x, -offset_y)
    
    data = np.array(final_img)
    alpha = data[:, :, 3]
    
    # PALETTE QUANTIZATION (K-Means)
    mask = alpha > 0
    visible_pixels = data[mask][:, :3] 
    
    kmeans = MiniBatchKMeans(n_clusters=colors, random_state=42, n_init="auto")
    kmeans.fit(visible_pixels)
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    data[mask, :3] = quantized_pixels
    
    # Save the test asset
    final_pil = Image.fromarray(data)
    final_pil.save(output_path, "PNG")

def main():
    os.makedirs("../results/final_assets/grid_tests", exist_ok=True)
    input_file = "../results/final_assets/cozy_girl_ULTIMATE_6.png"
    
    # We test the 4 major diagonal offsets (0 to 3) since it's a 4x4 block grid.
    offsets = [(0, 0), (1, 1), (2, 2), (3, 3)]
    
    for ox, oy in offsets:
        out_name = f"../results/final_assets/grid_tests/cozy_girl_offset_{ox}_{oy}.png"
        apply_offset_pipeline(input_file, out_name, ox, oy)
        
    print("\nGrid Offset Testing complete!")

if __name__ == "__main__":
    main()
